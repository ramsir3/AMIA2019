from auto_ml import Predictor
from tpot import TPOTClassifier

def runAutoML(df_train, df_test):
    column_descriptions = {
        'STAGE': 'output'
    }
    models = [
              'AdaBoostClassifier',
              'ExtraTreesClassifier',
              'GradientBoostingClassifier',
              'MiniBatchKMeans',
              'PassiveAggressiveClassifier',
              'Perceptron',
              'RandomForestClassifier',
              'RidgeClassifier',
              'SGDClassifier',
              'DeepLearningClassifier',
              'LGBMClassifier',
              'XGBClassifier',
             ]
    models = None
    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_train, model_names=models)
    return ml_predictor.score(df_test, df_test.STAGE)

def runTPot(X_train, X_test, y_train, y_test):
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline.py')