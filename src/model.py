import numpy as np
import cvxpy as cp


class Generator:
    def __init__(self, specifications: dict):

        self.name = specifications["name"]

        self.alpha = specifications["alpha"]
        self.beta = specifications["beta"]
        self.gamma = specifications["gamma"]

        self.max_power = specifications["max_power"]
        self.min_power = specifications["min_power"]

        self.ramp_up_rate = specifications["ramp_up_rate"]
        self.ramp_down_rate = specifications["ramp_down_rate"]

    def compute_cost(self, power: cp.Variable):

        cost = self.alpha * cp.power(power[0], 2) + self.beta * (power) + self.gamma

        return cost.sum()


class EconomicDispatchModel:
    def __init__(
        self, generators: list[Generator], horizon: int, lambdas: tuple[float, float]
    ):
        self.generators = generators
        self.num_generators = len(generators)

        self.horizon = horizon

        self.lambda_over = lambdas[0]
        self.lambda_under = lambdas[1]

        self.model = self.create_ed()

    def create_ed(self):
        # Parameters
        self.d = cp.Parameter(self.horizon)

        # Variables
        self.g = cp.Variable((self.num_generators, self.horizon), nonneg=True)
        self.d_over = cp.Variable(self.horizon, nonneg=True)
        self.d_under = cp.Variable(self.horizon, nonneg=True)

        # Objetive function
        objective = 0

        for index, generator in enumerate(self.generators):
            for t in range(self.horizon):
                objective += (
                    generator.alpha * cp.power(self.g[index, t], 2)
                    + generator.beta * (self.g[index, t])
                    + generator.gamma
                )

        objective += cp.sum(self.d_over) * self.lambda_over
        objective += cp.sum(self.d_under) * self.lambda_under

        # Constraints
        constraints = []

        # Constraint 1 (Conservation of energy):
        for t in range(self.horizon):
            constraints.append(
                self.g[:, t].sum() + self.d_over[t] - self.d_under[t] == self.d[t]
            )

        # Constraint 2 (Maximum and minimum power constraints):
        for index, generator in enumerate(self.generators):
            constraints.append(self.g[index, :] <= generator.max_power)
            constraints.append(self.g[index, :] >= generator.min_power)

        # Constraint 3 (Maximum and minimum ramp rates):
        for index, generator in enumerate(self.generators):
            for t in range(1, self.horizon):
                constraints.append(
                    cp.pos(self.g[index, t] - self.g[index, t - 1])
                    <= generator.ramp_up_rate
                )

                constraints.append(
                    cp.neg(self.g[index, t] - self.g[index, t - 1])
                    <= generator.ramp_down_rate
                )

        model = cp.Problem(cp.Minimize(objective), constraints)

        return model

    def solve_ed(self, demand: np.ndarray):

        self.d.value = demand

        return self.model.solve()
