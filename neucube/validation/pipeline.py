from neucube.training import STDP

class Pipeline():
    """
    Neucube pipeline consisting of a reservoir, a sampling method, and a classifier.

    Attributes:
        res_model: Instantiated reservoir object.
        sampling_method: Instantiated sampler.
        classifier: Classifier from sklearn.

    Methods:
        __init__(res_model, sampling_method, classifier):
            Initializes the pipeline with the given components.

        fit(X_train, y_train, train=False):
            Trains the pipeline using the provided input data and target labels.

        predict(X_test):
            Makes predictions using the trained pipeline on the given input data.
    """

    def __init__(self, res_model, sampling_method, classifier):
        """
        Initializes the pipeline with the provided components.

        Args:
            res_model: Instantiated reservoir object.
            sampling_method: Instantiated sampler.
            classifier: Classifier from sklearn.
        """
        self.res_model = res_model
        self.sampling_method = sampling_method
        self.classifier = classifier

    def fit(self, X_train, y_train, train=False, learning_rule=STDP(), verbose=False):
        """
        Trains the pipeline using the provided input data and target labels.

        Args:
            X_train: The input data for training.
            y_train: The target labels for training.
            learning_rule (LearningRule): The learning rule implementation to use for training.
            train: Optional boolean indicating if the reservoir should be trained or not.
                   Default is False.
            verbose (bool): Flag indicating whether to display progress during simulation.

        Returns:
            None
        """
        s_act = self.res_model.simulate(X_train, train=train, learning_rule=learning_rule, verbose=verbose)
        state = self.sampling_method.sample(s_act)
        self.classifier.fit(state, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained pipeline on the given input data.

        Args:
            X_test: The input data for prediction.

        Returns:
            pred: The predicted labels.
        """
        s_act = self.res_model.simulate(X_test, train=False, verbose=False)
        state = self.sampling_method.sample(s_act)
        self.state_test = state
        pred = self.classifier.predict(state)
        return pred