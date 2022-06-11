from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from threading import Thread, Lock
from contextlib import redirect_stdout
import tensorflow as tf
import numpy as np
import pickle
import copy
import io
from keras import backend as K

def windowed_range(n):
    labels = np.arange(n, dtype=np.float32) / float(n-1)
    thresholds = [(labels[i] + labels[i+1]) / 2. for i in range(n-1)]

    t_start = [0.]
    t_start.extend(thresholds)
    #print(f'Thresholds start:', t_start)

    t_stop = thresholds
    t_stop.append(1.)
    #print(f'Thresholds stop:', t_stop)
    
    for i in range(n):
        yield t_start[i], t_stop[i], i


def enumerate_predictions(predictions,  n_labels=2, fold=None, client=None):
    one_hot = np.eye(n_labels)

    # filter fold an client:
    if fold==None:
        y = predictions[:, :] if client==None else predictions[:, client]
    else:
        y = predictions[fold, :] if client==None else predictions[fold, client]

    # reshape to two columns:                        
    y = y.reshape((-1, 2))

    # filter NaN-values:
    y = y[~np.isnan(y).any(axis=1), :]

    for y_true, y_pred in y:
        # create labels:
        l = 0
        for start, stop, i in windowed_range(n_labels):
            if (y_true > start) & (y_true <= stop):
                l = i
                break

        # calculate error:
        e = np.abs(np.arange(n_labels, dtype=np.float32) - (float(n_labels-1) * y_pred))
        e = np.clip(e, 0., 1.)

        # Yield (y_true, y_pred)
        yield one_hot[l], 1-e


def stratified_split(data, n_splits, stratify=None, shuffle=False, random_state=None):
    result = None
    if (n_splits > 1): 
        i_split, i_remainder = train_test_split(
            range(len(data)),
            train_size=(1.0 / float(n_splits)),
            stratify=stratify,
            shuffle=shuffle,
            random_state=random_state
        )
        result = stratified_split(
            data[i_remainder],
            n_splits - 1, 
            stratify=None if np.any(stratify==None) else stratify[i_remainder],
            shuffle=shuffle,
            random_state=random_state
        )
        result.append(data[i_split])

    else:
        result = [data]

    return result


class ContinuousAUC(tf.keras.metrics.AUC):
    def __init__(self, num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, num_labels=2, from_logits=False):
        super().__init__(
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            name=name,
            dtype=dtype,
            #thresholds=thresholds,
            #multi_label=True,
            #num_labels=num_labels,
            #label_weights=None,
            from_logits=from_logits
        )

        self.n_labels = tf.constant(num_labels, dtype=tf.int32)
        self.t0 = np.empty(num_labels)
        self.t1 = np.empty(num_labels)
        for start, stop, i in windowed_range(self.n_labels):
            self.t0[i] = start
            self.t1[i] = stop
        self.t0 = tf.constant(self.t0, dtype=tf.float32)
        self.t1 = tf.constant(self.t1, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = tf.zeros_like(y_true, dtype=tf.int32)
        for i in range(self.n_labels):
            t0 = tf.gather(self.t0, [i])
            t1 = tf.gather(self.t1, [i])
            l = tf.where(tf.math.logical_and(y_true>t0, y_true<=t1), tf.cast(i, dtype=tf.int32), tf.cast(l, dtype=tf.int32))
        l = tf.keras.backend.one_hot(l, num_classes=self.n_labels)
        l = tf.reshape(l, (-1, self.n_labels))
        l = tf.slice(l, [0,1], [-1,-1])

        e = tf.math.abs(tf.range(self.n_labels, dtype=tf.float32) - (tf.cast(self.n_labels-1, dtype=tf.float32) * y_pred))
        e = tf.keras.backend.clip(e, 0., 1.)
        e = tf.reshape(e, (-1, self.n_labels))
        e = tf.slice(e, [0,1], [-1,-1])

        return super().update_state(
            l,
            1. - e,
            sample_weight=sample_weight
        )


class ContinuousPrecision(tf.keras.metrics.Precision):
    def __init__(self, top_k=None, label_id=None, name=None, dtype=None, num_labels=2):
        super().__init__(
            #thresholds=thresholds,
            top_k=top_k,
            class_id=label_id,
            name=name,
            dtype=dtype
        )

        self.n_labels = tf.constant(num_labels, dtype=tf.int32)
        self.t0 = np.empty(num_labels)
        self.t1 = np.empty(num_labels)
        for start, stop, i in windowed_range(self.n_labels):
            self.t0[i] = start
            self.t1[i] = stop
        self.t0 = tf.constant(self.t0, dtype=tf.float32)
        self.t1 = tf.constant(self.t1, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = tf.zeros_like(y_true, dtype=tf.int32)
        for i in range(self.n_labels):
            t0 = tf.gather(self.t0, [i])
            t1 = tf.gather(self.t1, [i])
            l = tf.where(tf.math.logical_and(y_true>t0, y_true<=t1), tf.cast(i, dtype=tf.int32), tf.cast(l, dtype=tf.int32))
        l = tf.keras.backend.one_hot(l, num_classes=self.n_labels)
        l = tf.reshape(l, (-1, self.n_labels))
        l = tf.slice(l, [0,1], [-1,-1])

        e = tf.math.abs(tf.range(self.n_labels, dtype=tf.float32) - (tf.cast(self.n_labels-1, dtype=tf.float32) * y_pred))
        e = tf.keras.backend.clip(e, 0., 1.)
        e = tf.reshape(e, (-1, self.n_labels))
        e = tf.slice(e, [0,1], [-1,-1])

        return super().update_state(
            l,
            1. - e,
            sample_weight=sample_weight
        )


class ContinuousRecall(tf.keras.metrics.Recall):
    def __init__(self, top_k=None, label_id=None, name=None, dtype=None, num_labels=2):
        super().__init__(
            #thresholds=thresholds,
            top_k=top_k,
            class_id=label_id,
            name=name,
            dtype=dtype
        )

        self.n_labels = tf.constant(num_labels, dtype=tf.int32)
        self.t0 = np.empty(num_labels)
        self.t1 = np.empty(num_labels)
        for start, stop, i in windowed_range(self.n_labels):
            self.t0[i] = start
            self.t1[i] = stop
        self.t0 = tf.constant(self.t0, dtype=tf.float32)
        self.t1 = tf.constant(self.t1, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = tf.zeros_like(y_true, dtype=tf.int32)
        for i in range(self.n_labels):
            t0 = tf.gather(self.t0, [i])
            t1 = tf.gather(self.t1, [i])
            l = tf.where(tf.math.logical_and(y_true>t0, y_true<=t1), tf.cast(i, dtype=tf.int32), tf.cast(l, dtype=tf.int32))
        l = tf.keras.backend.one_hot(l, num_classes=self.n_labels)
        l = tf.reshape(l, (-1, self.n_labels))
        l = tf.slice(l, [0,1], [-1,-1])

        e = tf.math.abs(tf.range(self.n_labels, dtype=tf.float32) - (tf.cast(self.n_labels-1, dtype=tf.float32) * y_pred))
        e = tf.keras.backend.clip(e, 0., 1.)
        e = tf.reshape(e, (-1, self.n_labels))
        e = tf.slice(e, [0,1], [-1,-1])

        return super().update_state(
            l,
            1. - e,
            sample_weight=sample_weight
        )


class Trainer:
    def __init__(self, vitals, labs, loss, metrics, output_signature, threaded=True, max_threads=2, random_state=None):
        '''Creates a new PipeBuilder-object.

        PARAMETERS
            vitals (Pandas Grouped DataFrame):  Vital values grouped by ICU-stay_id

            labs (Pandas Grouped DataFrame):    Lab values grouped by ICU-stay_id

            loss (tf.keras.losses):          
            
            Loss function for training

            metrics (tf.keras.metrics):         List of metrics used during training

            output_signature:                   tf-signature of the model in- and outputs
            
            threaded (bool):                    Wheter client-models are trained in parallel threads (default: True)
            
            random_state (int):                 Seed for the random generator (default: None)

        '''
        self.vitals = vitals
        self.labs = labs
        self.loss = loss
        self.metrics = metrics 
        self.output_signature = output_signature
        self.threaded = threaded
        self.max_threads = max_threads
        self.random_state = random_state
        
        self.max_vitals = np.max([v[1].shape[0] for v in vitals])
        self.max_labs = np.max([l[1].shape[0] for l in labs])

        self.lock = Lock()
        self.printing_active = True

        self.__init_normalization()
        self.__init_properties(1,1,1,1)

    def __init_properties(self, n_clients, n_folds, n_epochs, data_size):
        self.lock.acquire(True)

        # Results:
        self.labels = ['loss']
        self.labels.extend([m.name for m in self.metrics])
        self.train_scores = {l: np.full((n_folds, n_clients, n_epochs), np.NaN, dtype=float) for l in self.labels}
        self.valid_scores = {l: np.full((n_folds, n_clients, n_epochs), np.NaN, dtype=float) for l in self.labels}
        self.test_scores = {l: np.zeros(n_folds, dtype=float) for l in self.labels}
        self.predictions = np.full((n_folds, n_clients, int(np.ceil(float(data_size) / float(n_folds))), 2), np.NaN, dtype=float)

        # Weights:
        self.global_weights = None
        self.client_weights = {}

        # Console output:
        self.console_buffer = ['' for i in range(n_clients)]

        self.lock.release()

    def __init_normalization(self, icustays=[]):
        self.max_v = None
        self.max_l = None
        self.min_v = None
        self.min_l = None

        if len(icustays) > 0:
            # find min and max for vitals and labs:
            for X, y in self.icustays2data(icustays):
                act_max_v = np.max(X[0], axis=0)
                act_max_l = np.max(X[1], axis=0)

                act_min_v = np.min(np.where(X[0] < 0, float('inf'), X[0]), axis=0)
                act_min_l = np.min(np.where(X[1] < 0, float('inf'), X[1]), axis=0)

                self.max_v = act_max_v if self.max_v is None else np.maximum(act_max_v, self.max_v)
                self.max_l = act_max_l if self.max_l is None else np.maximum(act_max_l, self.max_l)
                self.min_v = act_min_v if self.min_v is None else np.minimum(act_min_v, self.min_v)
                self.min_l = act_min_l if self.min_l is None else np.minimum(act_min_l, self.min_l)
            
            self.max_v = tf.constant(self.max_v, dtype=tf.float64)
            self.max_l = tf.constant(self.max_l, dtype=tf.float64)
            self.min_v = tf.constant(self.min_v, dtype=tf.float64)
            self.min_l = tf.constant(self.min_l, dtype=tf.float64)

    def __split_test(self, icustays, n_folds=5, n_labels=2, shuffle=False):
        y = np.zeros_like(icustays[:,1], dtype=int)
        for start, stop, i in windowed_range(n_labels):
            y[np.where(np.logical_and(icustays[:,1]>start, icustays[:,1]<=stop))] = i

        # For each cross-validation-fold:
        fold = 0
        for i_train, i_test in StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=self.random_state).split(icustays[:,0], y):
            fold += 1

            yield i_train, i_test, fold

    def __split_valid(self, icustays, indices, n_clients=1, n_labels=2, shuffle=False, stratify=False):
        y = np.zeros_like(icustays[:,1], dtype=int)
        for start, stop, i in windowed_range(n_labels):
            y[np.where(np.logical_and(icustays[:,1]>start, icustays[:,1]<=stop))] = i

        # For each FL-client:
        client = 0
        for split in stratified_split(indices, n_clients, stratify=y[indices] if stratify else None, shuffle=shuffle, random_state=self.random_state):
            client += 1
            
            # Split validation data:
            i_train, i_valid = train_test_split(
                split,
                train_size=0.85,
                stratify=y[split] if stratify else None,
                shuffle=shuffle,
                random_state=self.random_state
            )

            yield i_train, i_valid, client

    def __run_model(self, model, client=1, fold=1, fl_round=-1, epochs=1, callbacks=None, data_train=None, data_valid=None, data_test=None):
        # Check whether FL is active:
        fl = fl_round >= 0

        # Fit model:
        history = model.fit(
            data_train,
            validation_data=None if fl else data_valid,
            callbacks=callbacks,
            epochs=epochs,
            verbose=1 if self.printing_active else 0
        )

        # Evaluate model:
        if self.printing_active:
                print('Evaluation:')
        scores = model.evaluate(
            data_valid if fl else data_test,
            verbose=1 if self.printing_active else 0
        )

        # Create label-prediction pairs:
        preds = np.full_like(self.predictions[fold-1, client-1], np.NaN, dtype=float)
        if not fl:
            i = 0
            for X, y in data_test:
                n = len(y)
                preds[i:i+n, 0] = y.numpy().reshape(n)
                preds[i:i+n, 1] = model.predict(X).reshape(n)
                i += n

        # Save results:
        self.lock.acquire(True)

        for i in range(len(self.labels)):
            l = self.labels[i]

            if fl:
                self.train_scores[l][fold-1, client-1, fl_round] = history.history[l][-1]
                self.valid_scores[l][fold-1, client-1, fl_round] = scores[i]

            else:
                self.train_scores[l][fold-1, client-1, :len(history.history[l])] = history.history[l]
                self.valid_scores[l][fold-1, client-1, :len(history.history['val_'+l])] = history.history['val_'+l]
                self.test_scores[l][fold-1] += scores[i]
        
        if not fl:
            self.predictions[fold-1, client-1, :, :] = preds

        if not self.printing_active:
            s = 'Scores: '
            for i in range(len(self.labels)):
                s += f'{self.labels[i]:s} = {scores[i]:.4f}; '
            self.enqueue_console(client, s)

        self.client_weights[client] = self.__get_model_weights(model)

        self.lock.release()

        del history
        del scores
        del preds
        del model

    def __run_threads(self, threads):
        self.printing_active = False

        active = []
        # Start threads:
        while len(active) <= self.max_threads and len(threads) > 0:
            t = threads.pop()
            active.append(t)
            t.start()

        # Wait for threads to finish:
        while len(active) > 0:
            t = active.pop(0)
            t.join()
            del t

            if len(threads) > 0:
                t = threads.pop()
                active.append(t)
                t.start()

        # Clear threads:  
        threads.clear()

        self.printing_active = True

        # Print output:
        self.flush_console()

    def __set_model_weights(self, model, weights):
        for i in range(len(weights)):
            model.layers[i].set_weights(weights[i])

    def __get_model_weights(self, model):
        return [layer.get_weights() for layer in model.layers]


    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(
                (self.train_scores, self.valid_scores, self.test_scores, self.predictions),
                file
            )


    def load(self, path):
        try:
            with open(path, 'rb') as file:
                self.train_scores, self.valid_scores, self.test_scores, self.predictions = pickle.load(
                    file
                )
        except:
            with open(path, 'rb') as file:
                self.train_scores, self.valid_scores, self.test_scores = pickle.load(
                    file
                )


    def plot_history(self, key, ax, x_step=2, client=None):
        '''Prints a specific metric from a list of tf.history objects.

        PARAMETERS
            key (string):       Name of the metric to be sampled

            ax (plt.axes):      Pyplot axes object which should be used for plotting

            x_step (int):       Label step of x-values

            client (any):       Dictionary key of the client to be sampled (default: None)
        '''

        # Get list of dictionaries:
        values_train = (self.train_scores[key][:, client-1] if client != None else self.train_scores[key][:, :])
        while len(values_train.shape) > 1:
            values_train = np.nanmean(values_train, axis=0) 
        n_train = values_train.shape[0]

        values_valid = (self.valid_scores[key][:, client-1] if client != None else self.valid_scores[key][:, :])
        while len(values_valid.shape) > 1:
            values_valid = np.nanmean(values_valid, axis=0)
        n_valid = values_valid.shape[0]

        ax.plot(np.arange(1,n_train+1), values_train, label='train')
        ax.plot(np.arange(1,n_valid+1), values_valid, label='valid')

        ax.set_xticks(np.arange(max(n_valid, n_train), step=x_step))
        ax.set_xlabel('epoch')
        ax.set_title(key)
        ax.legend()


    def enqueue_console(self, client, text):
        self.console_buffer[client-1] += text + '\n'


    def flush_console(self):
        for i in range(len(self.console_buffer)):
            print(self.console_buffer[i])
            self.console_buffer[i] = ''


    def icustays2data(self, icustays):
        '''Generator, that yields the data matching a list of ICU-stays.

        PARAMETERS
            icustays (np.array):    Array containing icustay-ids and labels


        YIELDS
            A tuple (X, y) of data X and label y
        '''

        flatten = tf.keras.layers.Flatten()

        for icustay, label in icustays:
            # Extract vitals:
            v = flatten(self.vitals.get_group(icustay).drop('icustay_id', axis=1).to_numpy())
            # Extract lab-values:
            l = flatten(self.labs.get_group(icustay).drop('icustay_id', axis=1).to_numpy())

            # Create label-tensor:
            y = tf.constant(label, dtype=tf.dtypes.float64, shape=1)

            yield (v, l), y


    def normalize(self, X, y):
        '''Performs normalizazion on each sample.

        PARAMETERS
            X (tensor): Data sample in the form of a tuple (vitals, labs)
            y (tensor): Data label


        RETURNS
            A tuple (X, y) of normalized data X and label y
        '''
        
        X = (
            tf.math.divide(tf.math.subtract(X[0], self.min_v), tf.math.subtract(self.max_v, self.min_v)), #Normalize vitals
            tf.math.divide(tf.math.subtract(X[1], self.min_l), tf.math.subtract(self.max_l, self.min_l))  #Normalize labs
        )
        
        return X, y


    def build_pipeline(self, icustays, batch_size=64, n_labels=2, oversample=False, weighted=False):
        '''Builds a data pipeline.

        PARAMETERS
            icustays (np.array):    Array containing icustay-ids and labels

            batch_size (int):       Batch size per thousand samples used for the datasets (default: 64)
            
            n_labels (int):         Number of bins used for oversampling and weighting (default: 2)

            oversample (bool):      Randomly oversamples data if True (default: False)

            weighted (bool):        Generates sample weights if True (default: False)


        RETURNS
            tf-dataset containing the icustays

        '''

        # Calculate class imbalance:
        if weighted:
            # Init weights:
            r = np.empty(n_labels, dtype=float)
            
            # Fill r and print used weights:
            print("\nSample weights per window:")
            for t0, t1, i in windowed_range(n_labels):
                r[i] = icustays.shape[0] / np.sum((icustays[:,1] >= t0) & (icustays[:,1] <= t1))
                print(f"{t0:.2f} < y < {t1:.2f}: {r[i]:.2f}")
            print()
        
        elif oversample:
            # Oversample minority-class:
            ros = RandomOverSampler(random_state=self.random_state)
            l = np.minimum(n_labels * icustays[:,1], n_labels-1).astype(int) 
            icustays,_ = ros.fit_resample(icustays, l)
            
            # Print assumed labels and their counts:
            print("\nSample counts per window:")
            for t0, t1, i in windowed_range(n_labels):
                print(f"{t0:.2f} < y < {t1:.2f}: {(l==i).sum():d}")
            print()
        
        # Create datasets:
        data = tf.data.Dataset.from_generator(
            self.icustays2data,
            args=[icustays],
            output_signature=self.output_signature
        ).cache()
        
        # Normalize data:
        data = data.map(self.normalize)
        
        # Add backpropagation weight to each sample:
        if weighted:
            data = data.map(
                lambda X, y: (X, y, tf.gather(r, tf.math.minimum(tf.cast(n_labels * y, dtype=tf.int64), n_labels-1)))
            )
        
        # Shuffle and batch data for each epoch:
        data = data.shuffle(5).padded_batch(int(batch_size), padding_values=tf.cast(-2., dtype=tf.float64))

        # Create prefetchable tf-dataset:
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return data


    def evaluate(self, model, icustays, n_folds=5, n_clients=1, n_labels=2, shuffle=False, oversample=False, weighted=False, stratify_clients=False):
        '''Evaluates a model.

        PARAMETERS
            model (tf.keras.model):     Model to evaluate

            icustays (np.array):        Array containing icustay-ids and labels

            n_folds (int):              Number of cross validation folds (default: 5)

            n_clients (int):            Number of local models (default: 1)

            n_labels (int):             Number of bins used for oversampling and weighting (default: 2)

            shuffle (bool):             Determines whether data is shuffeled before creating splits (default: False)

            oversample (bool):          Randomly oversamples data if True (default: False)

            weighted (bool):            Generates sample weights if True (default: False)
        '''

        # Init properties:
        self.__init_properties(n_clients, n_folds, 50, len(icustays))

        # Multithreading variables:
        use_threads = n_clients > 1 and self.threaded
        threads = []

        # Calculate batch size:
        batch_size = int(512/n_clients)

        # For each cross-validation-fold:
        for i_rest, i_test, fold in self.__split_test(icustays, n_folds=n_folds, n_labels=n_labels, shuffle=shuffle):

            # For each client:
            for i_train, i_valid, client in self.__split_valid(icustays, i_rest, n_clients=n_clients, n_labels=n_labels, shuffle=shuffle, stratify=stratify_clients):

                # New normalization bounds:
                self.__init_normalization(icustays[i_train])

                # Copy model:
                m = tf.keras.models.clone_model(model)

                # Compile model:
                m.compile(
                    loss=self.loss,
                    optimizer=tf.keras.optimizers.Adam(0.01),
                    metrics= self.metrics
                )

                # Print header:
                self.enqueue_console(client,
                    f'\n---------------------------------------------------------------------------' +
                    f'\nCross-validation iteration {fold:d}/{n_folds:d}; Client {client:d}/{n_clients:d}' +
                    f'\nTraining size = {len(i_train):d}; Validation size = {len(i_valid):d}; Test size = {len(i_test):d}' +
                    f'\nBatch size = {batch_size:d}' +
                    f'\n---------------------------------------------------------------------------'
                )

                # Build train- and validation pipelines:
                with redirect_stdout(io.StringIO()) as out:
                    data_train = self.build_pipeline(icustays[i_train], batch_size=batch_size, oversample=oversample, weighted=weighted, n_labels=n_labels)
                    data_valid = self.build_pipeline(icustays[i_valid], batch_size=batch_size)
                    data_test = self.build_pipeline(icustays[i_test], batch_size=batch_size)
                self.enqueue_console(client, '\n' + out.getvalue())
                del out

                # Callbacks:
                callbacks=[
                    tf.keras.callbacks.LearningRateScheduler(lambda epoch, eta: 0.5*eta if (epoch%5) == 0 and epoch > 0 else eta),
                    tf.keras.callbacks.EarlyStopping(
                        patience=20,
                        monitor='val_loss',
                        restore_best_weights=True
                    )
                ]

                if use_threads:
                    # Train each client in its own thread:
                    t = Thread(
                        target=self.__run_model, 
                        args=(m,),
                        kwargs={
                            'client': client,
                            'fold': fold,
                            'epochs': 50,
                            'callbacks': callbacks, 
                            'data_train': data_train,
                            'data_valid': data_valid,
                            'data_test': data_test
                        }
                    )
                    threads.append(t)

                else:
                    # Print output:
                    self.flush_console()

                    # Train each client in local thread:
                    self.__run_model(
                        m,
                        client=client,
                        fold=fold,
                        epochs=50,
                        callbacks=callbacks,
                        data_train=data_train,
                        data_valid=data_valid,
                        data_test=data_test
                    )

                    # Print output:
                    self.flush_console()

            if use_threads:
                self.__run_threads(threads)

        for key in self.test_scores:
            self.test_scores[key] /= n_clients


    def evaluateFL(self, model, icustays, n_folds=5, n_clients=1, n_rounds=50, n_labels=2, shuffle=False, oversample=False, weighted=False, stratify_clients=False):
        '''Evaluates a model with federated learning.

        PARAMETERS
            model (tf.keras.model): Model to evaluate

            icustays (np.array):    Array containing icustay-ids and labels

            n_folds (int):          Number of cross validation folds (default: 5)

            n_clients (int):        Number of local models (default: 1)

            n_rounds (int):         Number of FL-rounds (default: 50)

            n_labels (int):         Number of bins used for oversampling and weighting (default: 2)

            shuffle (bool):         Determines whether data is shuffeled before creating splits (default: False)

            oversample (bool):      Randomly oversamples data if True (default: False)

            weighted (bool):        Generates sample weights if True (default: False)
        '''

        # Init properties:
        self.__init_properties(n_clients, n_folds, n_rounds, len(icustays))

        # Multithreading variables:
        use_threads = n_clients > 1 and self.threaded
        threads = []

        # Calculate batch size:
        batch_size = int(512/n_clients)

        # For each cross-validation-fold:
        for i_rest, i_test, fold in self.__split_test(icustays, n_folds=n_folds, n_labels=n_labels, shuffle=shuffle):
            
            # New normalization bounds:
            self.__init_normalization(icustays[i_rest])

            # Build client data splits:
            clients = {}
            for i_train, i_valid, client in self.__split_valid(icustays, i_rest, n_clients=n_clients, n_labels=n_labels, shuffle=shuffle, stratify=stratify_clients):
                print(
                    f'\n---------------------------------------------------------------------------' +
                    f'\nCross-validation iteration {fold:d}/{n_folds:d}; Client {client:d}/{n_clients:d}' +
                    f'\nTraining size = {len(i_train):d}; Validation size = {len(i_valid):d}' +
                    f'\nBatch size = {batch_size:d}' +
                    f'\n---------------------------------------------------------------------------'
                )

                # Create datasets and model:
                clients[client] = {
                    'model':        tf.keras.models.clone_model(model),
                    'data_train':   self.build_pipeline(icustays[i_train], batch_size=batch_size, oversample=oversample, weighted=weighted, n_labels=n_labels), 
                    'data_valid':   self.build_pipeline(icustays[i_valid], batch_size=batch_size),
                    'n':            len(i_train)
                }

                # Compile model:
                clients[client]['model'].compile(
                    loss=self.loss,
                    optimizer=tf.keras.optimizers.Adam(0.01),
                    metrics=self.metrics
                )

            # Init global model weights:
            self.global_weights = self.__get_model_weights(clients[1]['model'])

            # For each FL-round:
            best_loss = (np.Inf, -1, None)
            for round in range(n_rounds):

                # For each FL-client:
                for client in clients:

                    # Set model weights:
                    self.__set_model_weights(clients[client]['model'], self.global_weights)

                    # Print header:
                    self.enqueue_console(client,
                        f'\n---------------------------------------------------------------------------' +
                        f'\nCross-validation iteration {fold:d}/{n_folds:d}; Client {client:d}/{n_clients:d}; Round {round+1:d}/{n_rounds:d}' +
                        f'\n---------------------------------------------------------------------------'
                    )

                    # Callbacks:
                    callbacks=[
                        tf.keras.callbacks.LearningRateScheduler(lambda epoch, eta: 0.5*eta if epoch == 0 and (round%5) == 0 and round > 0 else eta)
                    ]

                    if use_threads:
                        # Train each client in its own thread:
                        t = Thread(
                            target=self.__run_model,
                            args=(clients[client]['model'],),
                            kwargs={
                                'client': client,
                                'fold': fold,
                                'fl_round': round,
                                'epochs': 1,
                                'callbacks': callbacks, 
                                'data_train': clients[client]['data_train'],
                                'data_valid': clients[client]['data_valid']
                            }
                        )
                        threads.append(t)

                    else:
                        # Print output:
                        self.flush_console()

                        # Train each client in local thread:
                        self.__run_model(
                            clients[client]['model'],
                            client=client,
                            fold=fold,
                            fl_round=round,
                            epochs=1,
                            callbacks=callbacks, 
                            data_train=clients[client]['data_train'],
                            data_valid=clients[client]['data_valid']
                        )

                        # Print output:
                        self.flush_console()

                if use_threads:
                    self.__run_threads(threads)

                # Calculate average weights:
                self.global_weights = []
                n = np.sum([clients[client]['n'] for client in clients], dtype=float)
                for client in self.client_weights:
                    frac = float(clients[client]['n']) / n
                    print(f'Factor client {client:d}: {frac:.2f}')
                    for i in range(len(self.client_weights[client])):
                        if len(self.global_weights) <= i:
                            self.global_weights.append([frac * self.client_weights[client][i][j] for j in range(len(self.client_weights[client][i]))])
                        else:
                            self.global_weights[i] = [self.global_weights[i][j] + (frac * self.client_weights[client][i][j]) for j in range(len(self.global_weights[i]))]

                # Early stopping:
                val_loss = self.valid_scores['loss'][fold-1, :, round].mean()
                if val_loss < best_loss[0]:
                    best_loss = (val_loss, round, copy.deepcopy(self.global_weights))
                    print(f'\nEarly stopping [round {round+1:d}]: Best loss {val_loss:.2f} stored for {len(self.global_weights):d} layers')

                elif (round - best_loss[1]) >= 20:
                    print(f'\nEarly stopping [round {round+1:d}]: Stopping training (Best round: {best_loss[1]+1:d})')
                    break

            print(
                f'\n---------------------------------------------------------------------------' +
                f'\nCross-validation iteration {fold:d}/{n_folds:d}; Global Model' +
                f'\nTest size = {len(i_test):d}' +
                f'\nBatch size = {batch_size:d}' +
                f'\n---------------------------------------------------------------------------'
            )

            # Copy model:
            m = tf.keras.models.clone_model(model)

            # Compile global model:
            m.compile(
                loss=self.loss,
                metrics=self.metrics
            )

            # Set global model:
            self.__set_model_weights(m, best_loss[2])

            # Evaluate global model:
            data_test = self.build_pipeline(icustays[i_test])
            scores = m.evaluate(data_test, batch_size=batch_size)

            # Save scores:
            for i in range(len(self.labels)):
                self.test_scores[self.labels[i]][fold-1] = scores[i]

            # Create label-prediction pairs:
            i = 0
            for X, y in data_test:
                n = len(y)
                self.predictions[fold-1, 0, i:i+n, 0] = y.numpy().reshape(n)
                self.predictions[fold-1, 0, i:i+n, 1] = m.predict(X).reshape(n)
                i += n

            #clear memory:
            del data_test
            del clients
            del scores
            del m

        