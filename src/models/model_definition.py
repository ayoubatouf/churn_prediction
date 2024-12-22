import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from config.config import SEARCH_SPACE_JSON_PATH
from src.utils.file_io import load_config


config = load_config(SEARCH_SPACE_JSON_PATH)


def bayesian_search_ada(
    X_train: pd.DataFrame, y_train: pd.Series
) -> AdaBoostClassifier:

    base_estimator_type = config["base_estimator"]["type"]
    if base_estimator_type == "DecisionTreeClassifier":
        base_estimator = DecisionTreeClassifier(
            max_depth=config["base_estimator"]["max_depth"]
        )
    else:
        raise ValueError("Unsupported base estimator type.")

    param_space = {
        "n_estimators": Integer(
            config["param_space"]["n_estimators"]["min"],
            config["param_space"]["n_estimators"]["max"],
        ),
        "learning_rate": Real(
            config["param_space"]["learning_rate"]["min"],
            config["param_space"]["learning_rate"]["max"],
            config["param_space"]["learning_rate"]["distribution"],
        ),
    }

    ada_model = AdaBoostClassifier(
        base_estimator=base_estimator, random_state=config["random_state"]
    )

    bayes_search = BayesSearchCV(
        estimator=ada_model,
        search_spaces=param_space,
        n_iter=config["n_iter"],
        cv=config["cv"],
        random_state=config["random_state"],
        verbose=2,
        n_jobs=-1,
    )

    bayes_search.fit(X_train, y_train)

    best_ada_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    print(f"Best parameters found: {best_params}")

    return best_ada_model
