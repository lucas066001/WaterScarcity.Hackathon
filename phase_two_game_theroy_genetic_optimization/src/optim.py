import numpy as np
from types import MethodType
import random
from copy import deepcopy
import copy
import pandas as pd


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

    def select(self, population, scores, nb_best):
        """
        Select half of the best individuals from the population based on their scores.

        Returns:
            tab of inidividus.
        """

        top_n_indexes = (
            scores[scores["sat_ok"] == True]
            .sort_values(by="score")
            .head(nb_best)
            .index.tolist()
        )

        return [population[i] for i in top_n_indexes]

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

        adjusted_score = (
            self.econ_weight * economic_impact - self.ecol_weight * ecological_impact
        )

        # To avoid negative value propagation
        if (
            high_sat < 0
            or med_sat < 0
            or low_sat < 0
            or economic_impact < -0.5
            or ecological_impact > 1
        ):
            ok_satisfaction = 0.0

        print("----------------")
        print("ecological_impact", ecological_impact)
        print("economic_impact", economic_impact)
        print("high_sat", high_sat)
        print("med_sat", med_sat)
        print("low_sat", low_sat)
        print("adjusted_score", adjusted_score)
        print("ok_satisfaction", ok_satisfaction)
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
        filtered = [r for r in self.execution_results if r["ok_satisfaction"] == 1.0]

        # Trouver celui avec le meilleur (max) adjusted_score
        if len(filtered) > 0:
            best_ind = max(filtered, key=lambda r: r["adjusted_score"])
            return best_ind
        else:
            # Si personne ne respecte le critère on renvoit le moins pire
            filtered = [r for r in self.execution_results]
            best_ind = max(filtered, key=lambda r: r["adjusted_score"])
            return best_ind

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
            scored = pd.DataFrame(columns=["score", "sat_ok"])

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

                scored.loc[len(scored)] = [adjusted_score, ok_satisfaction]
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
                            if adjusted_score >= max(scored["score"])
                            else None
                        ),
                    }
                )

            best_score = scored[scored["sat_ok"] == True].sort_values(
                by="score", ascending=False
            )
            if len(best_score) > 0:
                print(
                    f"Génération {gen}, meilleur score : {best_score.iloc[0]["score"]}, respect {best_score.iloc[0]["sat_ok"]}"
                )
            else:
                print(
                    "Cette génération n'a produit aucun individu viable, tous les individus meurent et d'autres seront générés aléatoirement"
                )

            # Reproduction, on prend les meilleurs et on les mutent
            if self.n_gen > 1:

                nb_best_parent_to_keep = self.pop_size // self.p_best_parents

                selected = self.select(population, scored, nb_best_parent_to_keep)
                # S'il n'y a pas assez de bon éléments dans la génération, on ne reproduit que ceux qui respectent la priorité
                # D'autres nouveaux sont créés à la place
                missing_parent = nb_best_parent_to_keep - len(selected)

                children = []

                if missing_parent != nb_best_parent_to_keep:
                    if missing_parent > 0:
                        print("We add only missing parents")
                        for i in range(missing_parent):
                            selected.append(self.generate_individuals())

                    while len(children) < self.pop_size:
                        parent = random.choice(selected)
                        child = self.mutate(parent, gen)
                        children.append(child)
                else:
                    print("nobody is kept from this generation")
                    print("We resample all population")
                    for i in range(self.pop_size):
                        children.append(self.generate_individuals())

            population = children

        # On retourne le réssultat d'éxecution au complet
        return self.execution_results
