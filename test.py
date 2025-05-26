v1.1 -> linear gap between actors
        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2 * params["PG"]

v1.2 -> exponential gap between actors (leading to negative values for low prio actors)
        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - 1.8 ** params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2.5 ** params["PG"]


v1.3 -> exponential gap adjusted over 0 