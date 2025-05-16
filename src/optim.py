import numpy as np
from types import MethodType
import random
import itertools
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
        pop_size=20,
        n_gen=30,
        mutation_rate=0.2,
        econ_weight=1,
        ecol_weight=1,
    ):
        """
        Create an EvolutionnarySearch object to run a parameter exploration around policies.

        Args:
            simulation: configurated simulation wich will serve as evaluation protocol.
            pop_size: number of generated variants per generation.
            n_gen: number of generation.
            mutation_rate: the proportion of children that will differ from parents.
            econ_weight: how economic impact is considerated in adjusted_score consideration.
            ecol_weight: how ecologic impact is considerated in adjusted_score consideration.

        Returns:
            EvolutionnarySearch object ready to run
        """
        self.simulation = simulation
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.econ_weight = econ_weight
        self.ecol_weight = ecol_weight
        self.execution_results = []

    def generate_individuals(self):
        """
        Generate required params for both policies.

        Returns:
            dict: Dictionary containing parameters for both incentive and quota policies \n
            {
                "incentive_params":{...},\n
                "quota_params": {...}
            }
        """
        return {
            "incentive_params": {
                # fines params
                "avg_incomes_fine_factor": random.uniform(0, 2),
                "actors_priority_fine_factor": random.uniform(-1, 1),
                "stress_fine_factor": random.uniform(-1, 1),
                # subsidies params
                "avg_incomes_subsidy_factor": random.uniform(0, 2),
                "actors_priority_subsidy_factor": random.uniform(-1, 1),
                "stress_subsidy_factor": random.uniform(-1, 1),
            },
            "quota_params": {
                "avg_pump_quota_factor": random.uniform(0, 2),
                "actor_priority_factor": random.uniform(-1, 1),
                "crisis_level_factor": random.uniform(-1, 1),
            },
        }

    def mutate(self, ind):
        """
        Generate new individus based on a set.
        Use the mutation rate to determine if the new inidividu will be different from the parent.

        Returns:
            tab of inidividus.
        """
        ind = deepcopy(ind)
        for k in ind:
            if random.random() < self.mutation_rate:
                ind[k] += np.random.normal(0, 0.1)
        return ind

    def select(self, population, scores):
        """
        Select half of the best individuals from the population based on their scores.

        Returns:
            tab of inidividus.
        """

        return [ind for _, ind in sorted(zip(scores, population), key=lambda x: -x[0])][
            : self.pop_size // 2
        ]

    def make_quota_function(params):
        """
        Based on given parameters it generate a quota policy.

        Returns:
            func: Quota policy.
        """

        def custom_quota(self, crisis_level, actors_priority, avg_pump, DOE, DCR):
            quota = (
                params["avg_pump_factor"] * avg_pump
                + params["actor_priority_factor"] * actors_priority
                + params["crisis_level_factor"] * crisis_level
            )
            # Forcer des quotas réalistes (>= 0)
            return np.clip(quota, 0, avg_pump)

        return custom_quota

    def make_incentive_function(params):
        """
        Based on given parameters it generate an incentive policy.

        Returns:
            func: Incentive policy.
        """

        def custom_incentive(
            self,
            actions: np.ndarray,
            actors_priority: np.ndarray,
            avg_incomes: np.ndarray,
            water_pump: np.ndarray,
            avg_pump: np.ndarray,
            is_crisis: np.ndarray,
            water_flows: np.ndarray,
            quota: np.ndarray,
            DOE=15,
            DCR=10,
        ):
            fine = np.zeros(self.nb_actors)
            stress = is_crisis[-1]  # Current crisis level

            # If average income is negative, replace it with 0
            avg_incomes = np.where(avg_incomes < 0, 0, avg_incomes)

            for i in range(self.nb_actors):

                # Apply fine if exceeding quota during crisis
                if (water_pump[i] > quota[i]) and (stress >= 0):
                    fine[i] = params["avg_incomes_fine_factor"] * avg_incomes[i]
                    +(params["actors_priority_fine_factor"] * (1 + actors_priority[i]))
                    +(params["stress_fine_factor"] * stress)

                # Give subsidy according to priority actors who respect their quota
                elif (water_pump[i] <= quota[i]) and (stress >= 0):
                    fine[i] = -(
                        params["avg_incomes_subsidy_factor"] * avg_incomes[i]
                        + (
                            params["actors_priority_subsidy_factor"]
                            * (1 + actors_priority[i])
                        )
                        + (params["stress_subsidy_factor"] * stress)
                    )

            return fine

        return custom_incentive

    def score_fn(incentive_policy, quota_policy, simulation):
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
        ecological_impact, economic_impact = simulation.get_final_scores_scaled()

        adjusted_score = economic_impact - ecological_impact

        return adjusted_score, ecological_impact, economic_impact

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
        execution_results = []

        for gen in range(self.n_gen):
            scored = []
            # Pour tous les individus d'une génération, on évalue le score
            for ind in population:
                quota_policy = self.make_quota_function(ind["quota_params"])
                incentive_policy = self.make_quota_function(ind["incentive_params"])
                adjusted_score, ecological_impact, economic_impact = self.score_fn(
                    incentive_policy, quota_policy, self.simulation
                )
                scored.append(adjusted_score)
                execution_results.append(
                    {
                        "generation": gen,
                        "quota_params": ind["quota_params"],
                        "incentive_params": ind["incentive_params"],
                        "ecological_impact": ecological_impact,
                        "economic_impact": economic_impact,
                        "adjusted_score": adjusted_score,
                    }
                )

            best_score = max(scored)
            print(f"Génération {gen}, meilleur score : {best_score:.4f}")

            # Reproduction, on prend la moitié des meilleurs et on les mutent
            selected = self.select(population, scored)
            children = []
            while len(children) < self.pop_size:
                parent = random.choice(selected)
                child = self.mutate(parent)
                children.append(child)

            population = children

        # On retourne le réssultat d'éxecution au complet
        return execution_results


from scipy.ndimage import gaussian_filter1d


from scipy.ndimage import gaussian_filter1d


def custom_incentive_policy(
    self,
    actions: np.ndarray,
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
) -> np.ndarray:

    # Normalisation afin de pouvoir justifier des diffénreces de poids entre les variables
    def min_max_normalize(array, force_min=None, force_max=None):
        if force_min is not None and force_max is not None:
            min_val = force_min
            max_val = force_max
        else:
            min_val = np.min(array)
            max_val = np.max(array)

        if max_val == min_val:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)

    # Initialisation du vecteur de prénalités / subventions
    fine = np.zeros(self.nb_actors)

    # On applique la normalisation
    actors_priority_norm = min_max_normalize(actors_priority)
    is_crisis_norm = min_max_normalize(is_crisis, force_min=-1, force_max=2)

    # Grâce à l'étude de correlation de la phase 1 on sait que la moyenne mobile
    # Gaussien est efficace pour prévenir des tendances futures
    def last_4_gaussian_avg(
        arr: np.ndarray, window: int = 4, sigma: float = 1.0
    ) -> float:
        if len(arr) < window:
            raise ValueError(f"Le tableau doit contenir au moins {window} éléments.")

        last_values = arr[-window:]
        weights = np.ones(window)
        gauss_weights = gaussian_filter1d(weights, sigma=sigma)
        return np.average(last_values, weights=gauss_weights)

    if len(water_flows) >= 5:
        water_flow_gauss = last_4_gaussian_avg(water_flows)
        water_flow_gauss_lag_1w = last_4_gaussian_avg(water_flows[:-1])
        # On créé un indicateur de tendance de l'eau
        # Comme on sait que cette tendance a tendance à s'inverser vis-à-vis des futurs flux d'eau
        # On pourra s'en servir pour anticiper des comportements et ajuster les pénalités
        water_flow_gauss_evol_slope = water_flow_gauss_lag_1w - water_flow_gauss
        # On divise cette valeur afin qu'elle soit dans un ordre de grandeur de 1
        # Cette valeur est arbitraire mais n'ayant pas de données comparable pour la mettre à l'échelle
        # Ce sera donc une valeur à régler en fonction de la station et des tendances hydrologiques observées
        adjusted_water_flow_gauss_evol_slope = water_flow_gauss_evol_slope / 15

    # On calcule la distance moyenne du flux d'eau par rapport au DCR
    # Cela permettra de savoir si nous sommes en période où la vigilance est nécessaire
    mean_water_flow = np.mean(water_flows)
    average_mwf_dcr_distance = mean_water_flow - DCR
    # On calcule si on est sur une distance moyenne du DCR qui est critique ou habituelle par rapport à la distance moyenne
    adjusted_mwf_dcr_distance = (water_flows[-1] - average_mwf_dcr_distance) / (
        average_mwf_dcr_distance
    )
    # Comme cette valeur peut varier entre des valeurs négatives et positives ~[-8,8]
    # On divise par 10 afin de rester dans une intervalle avec un ordre de grandeur de 1
    # Cet indicateur permettra d'anticiper les amendes et subventions avant d'atteindre les seuiles critiques
    adjusted_mwf_dcr_distance = adjusted_mwf_dcr_distance / 10

    # Une fois tous les indicateurs calculés on s'en sert pour déterminer les pénalités et subventions

    return fine


def tuned_incentive_policy(
    self,
    actions: np.ndarray,
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
    params: dict = None,
) -> np.ndarray:

    # Normalisation afin de pouvoir justifier des diffénreces de poids entre les variables
    def min_max_normalize(array, force_min=None, force_max=None):
        if force_min is not None and force_max is not None:
            min_val = force_min
            max_val = force_max
        else:
            min_val = np.min(array)
            max_val = np.max(array)

        if max_val == min_val:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)

    # Initialisation du vecteur de prénalités / subventions
    fine = np.zeros(self.nb_actors)

    # On applique la normalisation plus une petite valeur pour éviter les multiplications par 0
    EPSILON = 5e-2
    actors_priority_norm = min_max_normalize(actors_priority) + EPSILON
    is_crisis_norm = min_max_normalize(is_crisis, force_min=-1, force_max=2) + EPSILON
    # Niveau de crise actuel normalisé
    current_stress_level_norm = is_crisis_norm[-1]

    # Grâce à l'étude de correlation de la phase 1 on sait que la moyenne mobile
    # Gaussien est efficace pour prévenir des tendances futures
    def last_4_gaussian_avg(
        arr: np.ndarray, window: int = 4, sigma: float = 1.0
    ) -> float:
        if len(arr) < window:
            raise ValueError(f"Le tableau doit contenir au moins {window} éléments.")

        last_values = arr[-window:]
        weights = np.ones(window)
        gauss_weights = gaussian_filter1d(weights, sigma=sigma)
        return np.average(last_values, weights=gauss_weights)

    if len(water_flows) >= 5:
        water_flow_gauss = last_4_gaussian_avg(water_flows)
        water_flow_gauss_lag_1w = last_4_gaussian_avg(water_flows[:-1])
        # On créé un indicateur de tendance de l'eau
        # Comme on sait que cette tendance a tendance à s'inverser vis-à-vis des futurs flux d'eau
        # On pourra s'en servir pour anticiper des comportements et ajuster les pénalités
        water_flow_gauss_evol_slope = water_flow_gauss_lag_1w - water_flow_gauss
        # On divise cette valeur afin qu'elle soit dans un ordre de grandeur de 1
        # Cette valeur est arbitraire mais n'ayant pas de données comparable pour la mettre à l'échelle
        # Ce sera donc une valeur à régler en fonction de la station et des tendances hydrologiques observées
        adjusted_water_flow_gauss_evol_slope = water_flow_gauss_evol_slope / 15
    else:
        # Si on n'a pas assez de données pour calculer la tendance de l'eau on met 0 pour annuler son impact
        adjusted_water_flow_gauss_evol_slope = 0

    # On calcule la distance moyenne du flux d'eau par rapport au DCR
    # Cela permettra de savoir si nous sommes en période où la vigilance est nécessaire
    mean_water_flow = np.mean(water_flows)
    average_mwf_dcr_distance = mean_water_flow - DCR
    # On calcule si on est sur une distance moyenne du DCR qui est critique ou habituelle par rapport à la distance moyenne
    adjusted_mwf_dcr_distance = (water_flows[-1] - average_mwf_dcr_distance) / (
        average_mwf_dcr_distance
    )
    # Comme cette valeur peut varier entre des valeurs négatives et positives ~[-8,8]
    # On divise par 10 afin de rester dans une intervalle avec un ordre de grandeur de 1
    # Cet indicateur permettra d'anticiper les amendes et subventions avant d'atteindre les seuiles critiques
    adjusted_mwf_dcr_distance = adjusted_mwf_dcr_distance / 10

    # Cette valeur est la valeur indiquant que tout le monde pompe à son habitude et ne risque pas de passer sous le DCR
    # Il ne faut donc pas passer en dessous
    critical_overall_demand_treshold = np.sum(avg_pump) + DCR

    # Une fois tous les indicateurs calculés on s'en sert pour déterminer les pénalités et subventions
    # Objectifs :
    # - Augmenter coop en temps de crise -> Diminuer l'impact écologique
    # - Baisser la coop en temps normal -> Augmenter l'impact économique car constaté
    # qu'une trop grande coopération est néfaste en temps normal, surtout en configuration de prédictions biasées
    for i in range(self.nb_actors):

        # Définition des politiques
        CRISIS_FINE = (
            (avg_incomes[i] * params["cf_avg_incomes"])
            * (
                params["cf_actors_priority"] * (1 / actors_priority_norm[i])
            )  # Plus la priorité est grande moins la pénalité est élvée
            * (
                params["cf_curr_stress"] * (current_stress_level_norm)
            )  # Plus le stress est grand plus la pénalité est élvée
        )
        CRISIS_SUBSIDY = -(
            (avg_incomes[i] * params["cs_avg_incomes"])
            * (
                params["cs_actors_priority"] * (actors_priority_norm[i])
            )  # Plus la priorité est grande plus la subvention est élvée
            * (
                params["cs_curr_stress"] * current_stress_level_norm
            )  # Plus le stress est grand plus la subvention est élvée
        )

        # Point d'attention car si tout le monde pompe à son habitude
        # On peut passer sous le DCR
        # On commence à amorcer la politique de coopération
        # if water_flows[-1] <= critical_overall_demand_treshold:
        #     # On anticipe les mouvements des flux futurs en prenant en compte
        #     # L'inverse tendance de la moyenne mobile
        #     # Car lors de l'étude de corrélation on a vu que la tendance de la moyenne mobile
        #     # était inversement proportionnelle aux flux futurs

        #     ANTICIPATED_CRISIS_FINE = CRISIS_FINE * (
        #         params["pf_wf_slope"] * adjusted_water_flow_gauss_evol_slope
        #     )  # Plus la tendance de l'eau est à la hausse plus la pénalité est élevée

        #     ANTICIPATED_NONE_CRISIS_FINE = CRISIS_FINE * -(
        #         params["pf_wf_slope"] * adjusted_water_flow_gauss_evol_slope
        #     )  # Plus la tendance de l'eau est à la baisse plus la pénalité est faible

        #     # Non Coop
        #     if water_pump[i] > quota[i]:
        #         if adjusted_water_flow_gauss_evol_slope >= 0:
        #             # Potentiel crise à venir car hausse passée
        #             fine[i] = ANTICIPATED_CRISIS_FINE
        #         else:
        #             # Potentiel recouvrement car baisse passée
        #             fine[i] = ANTICIPATED_NONE_CRISIS_FINE
        #     # Coop
        #     else:
        #         if adjusted_water_flow_gauss_evol_slope >= 0:
        #             # Potentiel crise à venir car hausse passée
        #             fine[i] = -ANTICIPATED_CRISIS_FINE
        #         else:
        #             # Potentiel recouvrement car baisse passée
        #             fine[i] = -ANTICIPATED_NONE_CRISIS_FINE

        # Politique en temps de crise
        if current_stress_level_norm >= 1 + EPSILON:
            # Policy for actors who exceed their quota
            if water_pump[i] > quota[i]:
                fine[i] = CRISIS_FINE
            # Policy for actors who respect their quota
            elif water_pump[i] <= quota[i]:
                fine[i] = CRISIS_SUBSIDY

        # Non-crisis case
        else:
            # En temps normal on veut que les acteurs coopèrent moins
            if water_pump[i] > quota[i]:
                fine[i] = CRISIS_SUBSIDY
    if fine.max() != 0 and fine.min() != 0 and False:
        print(f"------------------- BEGIN Turn -------------------")
        print(f"max fine : {fine.max()}")
        print(f"min fine : {fine.min()}")
        print(f"current_stress_level_norm : {current_stress_level_norm}")
        print(
            f"adjusted_water_flow_gauss_evol_slope : {adjusted_water_flow_gauss_evol_slope}"
        )
        print(f"    params[cf_avg_incomes]: {params["cf_avg_incomes"]}")
        print(f"    params[cf_actors_priority]: {params["cf_actors_priority"]}")
        print(f"    params[cf_curr_stress]: {params["cf_curr_stress"]}")
        print(f"    params[cs_avg_incomes]: {params["cs_avg_incomes"]}")
        print(f"    params[cs_actors_priority]: {params["cs_actors_priority"]}")
        print(f"    params[cs_curr_stress]: {params["cs_curr_stress"]}")
        print(f"    params[pf_wf_slope]: {params["pf_wf_slope"]}")

        print(f"------------------- END Turn -------------------")
        import time

        time.sleep(0.5)
    return fine
