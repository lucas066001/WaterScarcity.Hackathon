import numpy as np
from types import MethodType
import random
from copy import deepcopy
import copy


class CustomRandomSearch:
    def __init__(
        self,
        simulation,
        custom_incentive_policy,
        custom_quota_policy,
        incentive_policy_params: dict,
        quota_policy_params: dict,
        max_iterations: int,
    ):
        """
        :param simulation: instance de la simulation à exécuter
        :param custom_incentive_policy: fonction d'incitation personnalisée
        :param custom_quota_policy: fonction de quota personnalisée
        :param incentive_policy_params: dict des hyperparamètres à tester (ex: {"NEGATIVE_POLICY_STRENGTH": [...], "POSITIVE_POLICY_STRENGTH": [...]})
        :param quota_policy_params: dict des hyperparamètres à tester (ex: {"ESTIMATION_FACTOR": [...], "NORMAL_GROWTH_FACTOR": [...]})
        :param max_iterations: nombre maximum d'itérations à tester
        """
        self.simulation = simulation
        self.custom_incentive_policy = custom_incentive_policy
        self.custom_quota_policy = custom_quota_policy
        self.incentive_policy_params = incentive_policy_params
        self.quota_policy_params = quota_policy_params
        self.max_iterations = max_iterations
        self.execution_results = []
        self._tested_combinations = set()

    def _random_combination(self, policy_params):
        keys = list(policy_params.keys())
        while True:
            combo = {k: float(random.choice(policy_params[k])) for k in keys}
            combo_tuple = tuple(sorted(combo.items()))
            if combo_tuple not in self._tested_combinations:
                self._tested_combinations.add(combo_tuple)
                return combo

    def run(self):
        for _ in range(self.max_iterations):
            print(f"Iteration {_+1}/{self.max_iterations}")
            # Obtenir une combinaison unique
            incentive_params = self._random_combination(self.incentive_policy_params)
            quota_params = self._random_combination(self.quota_policy_params)

            def incentive_policy(sim, *args, **kwargs):
                return self.custom_incentive_policy(
                    sim, *args, **kwargs, **incentive_params
                )

            def quota_policy(sim, *args, **kwargs):
                return self.custom_quota_policy(sim, *args, **kwargs, **quota_params)

            # Lier les politiques
            self.simulation.incentive_policy = MethodType(
                incentive_policy, self.simulation
            )
            self.simulation.compute_actor_quota = MethodType(
                quota_policy, self.simulation
            )

            self.simulation.run_simulation()
            ecological_impact, economic_impact = (
                self.simulation.get_final_scores_scaled()
            )

            # Enregistrement
            self.execution_results.append(
                {
                    "incentive_params": incentive_params,
                    "quota_params": quota_params,
                    "ecological_impact": ecological_impact,
                    "economic_impact": economic_impact,
                    # "simulation": copy.deepcopy(self.simulation)
                }
            )

    def get_best_result(self):
        # Normalisation des impacts
        eco_impacts = np.array([r["economic_impact"] for r in self.execution_results])
        ecol_impacts = np.array(
            [r["ecological_impact"] for r in self.execution_results]
        )

        # Normalisation [0, 1]
        eco_norm = (eco_impacts - eco_impacts.min()) / (
            eco_impacts.max() - eco_impacts.min()
        )
        ecol_norm = (ecol_impacts - ecol_impacts.min()) / (
            ecol_impacts.max() - ecol_impacts.min()
        )

        # Score combiné (éco haut, écolo bas)
        scores = eco_norm - ecol_norm

        # Meilleur score
        best_idx = np.argmax(scores)
        return self.execution_results[best_idx]


class EvolutionnarySearch:
    def __init__(
        self,
        simulation,
        p_best_parents=2,
        pop_size=20,
        n_gen=30,
        mutation_rate=0.2,
        initial_mutation_spread=0.1,
        econ_weight=1,
        ecol_weight=1,
        score_metric="eco",
    ):
        """
        Create an EvolutionnarySearch object to run a parameter exploration around policies.

        Args:
            simulation: configurated simulation wich will serve as evaluation protocol.
            p_best: proportion of best individus to select as parent for next generation : total_ind // p_best.
            pop_size: number of generated variants per generation.
            n_gen: number of generation.
            mutation_rate: the proportion of children that will differ from parents.
            initial_mutation_spread: how much childrens will differ from parents (decreasing along generation).
            econ_weight: how economic impact is considerated in adjusted_score consideration.
            ecol_weight: how ecologic impact is considerated in adjusted_score consideration.

        Returns:
            EvolutionnarySearch object ready to run
        """
        self.simulation = simulation
        self.p_best_parents = p_best_parents
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.initial_mutation_spread = initial_mutation_spread
        self.econ_weight = econ_weight
        self.ecol_weight = ecol_weight
        self.execution_results = []
        self.score_metric = score_metric
        self.make_quota_function = MethodType(self._default_make_quota_function, self)
        self.generate_individuals = MethodType(self._default_generate_individuals, self)
        self.make_incentive_function = MethodType(
            self._default_make_incentive_function, self
        )

    def mutate(self, ind, gen):
        """
        Mutate parameters of an individual. Operates on both 'quota_params' and 'incentive_params'.

        Args:
            ind: individu to mutate.
            gen: generation number, wich will impact the strengh of mutation.

        Returns:
            dict: mutated individual
        """
        ind = deepcopy(ind)

        current_mutation_spread = self.initial_mutation_spread * (
            (self.n_gen - gen) / self.n_gen
        )
        print(f"current_mutation_spread : {current_mutation_spread}")

        for param_key in ind:
            if random.random() < self.mutation_rate:
                ind[param_key] *= np.random.normal(1, current_mutation_spread)

        return ind

    def select(self, population, scores):
        """
        Select half of the best individuals from the population based on their scores.

        Returns:
            tab of inidividus.
        """

        return [ind for _, ind in sorted(zip(scores, population), key=lambda x: -x[0])][
            : self.pop_size // self.p_best_parents
        ]

    def _default_make_quota_function(self, params):
        raise ValueError(
            "You need to set the EvolutionnarySearch.make_quota_function = MethodType(self.your_implementation, self)"
        )

    def _default_make_incentive_function(self, params):
        raise ValueError(
            "You need to set the EvolutionnarySearch.make_incentive_function = MethodType(self.your_implementation, self)"
        )

    def _default_generate_individuals(self, params):
        raise ValueError(
            "You need to set the EvolutionnarySearch.generate_individuals = MethodType(self.your_implementation, self)"
        )

    def score_fn(self, incentive_policy, quota_policy, simulation):
        """
        Run a simulation based on give policy and compute scores.

        Returns:
            Tuple:  - adjusted_score = economic_impact - ecological_impact.\n
                    - ecological_impact = ecological impact of the simulation.\n
                    - economic_impact = economic impact of the simulation.
        """
        # Lier les politiques
        simulation.incentive_policy = MethodType(incentive_policy, simulation)
        simulation.compute_actor_quota = MethodType(quota_policy, simulation)

        simulation.run_simulation()

        (
            ecological_impact,
            economic_impact,
            high_sat,
            med_sat,
            low_sat,
            ok_satisfaction,
        ) = simulation.get_final_scores_scaled_alt()

        if self.score_metric == "priority_order":
            adjusted_score = (high_sat - med_sat) + (med_sat - low_sat)
        else:
            # If satisfaction ok we keep the score intact so it gets an advantage
            adjusted_score = (
                self.econ_weight * economic_impact
                - self.ecol_weight * ecological_impact
            )

            if not ok_satisfaction:
                adjusted_score -= 100
        print("----------------")
        print("ecological_impact", ecological_impact)
        print("economic_impact", economic_impact)
        print("high_sat", high_sat)
        print("med_sat", med_sat)
        print("low_sat", low_sat)
        print("adjusted_score", adjusted_score)
        print("----------------")
        return (
            adjusted_score,
            ecological_impact,
            economic_impact,
            ok_satisfaction,
            high_sat,
            med_sat,
            low_sat,
        )

    def get_best_result(self):
        adjusted_score = np.array([r["adjusted_score"] for r in self.execution_results])

        # Meilleur score
        best_idx = np.argmax(adjusted_score)
        return self.execution_results[best_idx]

    def run_search(self):
        """
        Run all simulations based on given search parameters.

        Returns:
            tab:
                - **dict**
                    - generation = generation number.
                    - quota_params = **dict** -> quota params.
                    - incentive_params = **dict** -> incentive params.
                    - ecological_impact = economic impact of the simulation.
                    - economic_impact = economic impact of the simulation.
                    - adjusted_score = economic impact of the simulation.
        """
        population = [self.generate_individuals() for _ in range(self.pop_size)]

        for gen in range(self.n_gen):
            scored = [-999]
            # Pour tous les individus d'une génération, on évalue le score
            it = 0
            for ind in population:
                it = it + 1
                print(f"iteration gen {gen} : {it}")
                quota_policy = self.make_quota_function(ind)
                incentive_policy = self.make_incentive_function(ind)
                (
                    adjusted_score,
                    ecological_impact,
                    economic_impact,
                    ok_satisfaction,
                    high_sat,
                    med_sat,
                    low_sat,
                ) = self.score_fn(incentive_policy, quota_policy, self.simulation)

                self.execution_results.append(
                    {
                        "generation": gen,
                        "params": ind,
                        "ecological_impact": ecological_impact,
                        "economic_impact": economic_impact,
                        "adjusted_score": adjusted_score,
                        "ok_satisfaction": ok_satisfaction,
                        "high_sat": high_sat,
                        "med_sat": med_sat,
                        "low_sat": low_sat,
                        "simulation": (
                            copy.deepcopy(self.simulation)
                            if adjusted_score >= max(scored)
                            else None
                        ),
                    }
                )
                scored.append(adjusted_score)

            best_score = max(scored)
            print(f"Génération {gen}, meilleur score : {best_score:.4f}")

            # Reproduction, on prend la moitié des meilleurs et on les mutent
            if self.n_gen > 1:
                selected = self.select(population, scored)
                children = []
                while len(children) < self.pop_size:
                    parent = random.choice(selected)
                    child = self.mutate(parent, gen)
                    children.append(child)

            population = children

        # On retourne le réssultat d'éxecution au complet
        return self.execution_results
