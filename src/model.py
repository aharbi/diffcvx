import numpy as np
import cvxpy as cp


class Generator:
    def __init__(self, specifications: dict):
        """Electrical generator object. Each generator has specifications that must be
        defined in a dictionary. The parameters are the following:
        name: name of the generator.
        alpha: coefficient of the quadratic term in the cost function.
        beta: coefficient of the linear term in the cost function.
        gamma: coefficient of the bias in the cost function.
        max_power: maximum power of the generator.
        min_power: minimum power of the generator.
        ramping_rate: ramping rate (i.e., the rate of change of power between time steps) of the generator.

        Args:
        specifications (dict): a dictionary of the specifications of the generator.
        """

        self.name = specifications["name"]

        self.alpha = specifications["alpha"]
        self.beta = specifications["beta"]
        self.gamma = specifications["gamma"]

        self.max_power = specifications["max_power"]
        self.min_power = specifications["min_power"]

        self.ramp_rate = specifications["ramp_rate"]

    def compute_cost(self, power: np.ndarray):
        """Computes the cost of a generator given a generation schedule.

        Args:
            power (np.ndarray): a vector of the generation schedule.

        Returns:
            float: cost of the generator.
        """

        cost = self.alpha * power**2 + self.beta * power + self.gamma

        return cost.sum()


class EconomicDispatchModel:
    def __init__(
        self,
        generators: list[Generator],
        horizon: int,
        lambda_plus: float = 1000,
        lambda_minus: float = 5000,
        relax_ramping: bool = False,
    ):
        """Economic dispatch problem object. The objective function is to minimize the
        cost of operating a set of generators to meet future electricity demand. Each
        generator is modeled with a quadratic cost function. The decision variables are
        the generation output of the generators for each time step. Also, slack variables are
        used to ensure that the problem is always feasible. A linear penalty is used to
        model the cost of using the slack variables. Each generator has maximum and
        minimum power constraints, as well as a maximum ramping rate constraint.

        Args:
        generators (list[Generator]): Generator objects for the power system.
        horizon (int): length of the planning horizon.
        lambda_plus (float, optional): cost of a positive mismatch for the slack variables. Defaults to 1000.
        lambda_minus (float, optional): cost of a negative mismatch for the slack variables. Defaults to 5000.
        relax_ramping (bool, optional): whether to include the ramping rate constraint or not. Defaults to False.
        """
        self.generators = generators
        self.num_generators = len(generators)

        self.horizon = horizon

        self.lambda_plus = lambda_plus
        self.lambda_minus = lambda_minus

        self.relax_ramping = relax_ramping

        self.model = self.create_ed()

    def create_ed(self):
        """Creates an instance of an economic dispatch optimization problem.

        Returns:
            cvxpy.Problem: a cvxpy problem instance of the economic dispatch formulation.
        """
        # Parameters
        self.d = cp.Parameter(self.horizon)

        # Variables
        self.g = cp.Variable((self.num_generators, self.horizon), nonneg=True)
        self.s_plus = cp.Variable(self.horizon, nonneg=True)
        self.s_minus = cp.Variable(self.horizon, nonneg=True)

        # Objetive function
        objective = 0

        for index, generator in enumerate(self.generators):
            for t in range(self.horizon):
                objective += (
                    generator.alpha * cp.power(self.g[index, t], 2)
                    + generator.beta * (self.g[index, t])
                    + generator.gamma
                )

        objective += cp.sum(self.s_plus) * self.lambda_plus
        objective += cp.sum(self.s_minus) * self.lambda_minus

        # Constraints
        constraints = []

        # Constraint 1 (Conservation of energy):
        for t in range(self.horizon):
            constraints.append(
                self.g[:, t].sum() + self.s_plus[t] - self.s_minus[t] >= self.d[t]
            )

        # Constraint 2 (Maximum and minimum power constraints):
        for index, generator in enumerate(self.generators):
            constraints.append(self.g[index, :] <= generator.max_power)
            constraints.append(self.g[index, :] >= generator.min_power)

        # Constraint 3 (Maximum and minimum ramp rates):
        if self.relax_ramping == False:
            for index, generator in enumerate(self.generators):
                for t in range(1, self.horizon):
                    constraints.append(
                        cp.abs(self.g[index, t] - self.g[index, t - 1])
                        <= generator.ramp_rate
                    )

        model = cp.Problem(cp.Minimize(objective), constraints)

        return model

    def solve_ed(self, demand: np.ndarray):
        """Solves an instance of the economic dispatch problem given an electricity
        demand time-series.

        Args:
            demand (np.ndarray): a vector of electricity demand of length self.horizon.

        Returns:
            float: optimal objective function value of the problem.
        """

        self.d.value = demand

        return self.model.solve()
