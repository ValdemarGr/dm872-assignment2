from numpy import *
import matplotlib
import matplotlib.pyplot as plt



class data:
    cust = 8 #Number of customers
    m = 3 #Maximum number of vehicles
    Q = 10 #Maximum vehicle capacity

    demand = array([4,  3,  3,   3,   3,  4,  3,  3  ]) #demand per customer
    twStart = array([0,  6,  8,  4,   6,   5,  6,  5,  4,  0]) #earlierst delivery time
    twEnd   = array([24, 6,  16, 20,  6,  19, 18, 19, 6, 24]) #latest delivery time


    # Travel cost matrix
    cost = array([
        [0,  7,  5,  3,  3,  4,  5,  4,  3,  0],
        [7,  0,  3,  5,  4, 11, 12, 10, 10,  7],
        [5,  3,  0,  5,  2,  9,  9,  8,  9,  5],
        [3,  5,  5,  0,  4,  6,  8,  7,  5,  3],
        [3,  4,  2,  4,  0,  6,  7,  6,  6,  3],
        [4, 11,  9,  6,  6,  0,  2,  2,  2,  4],
        [5, 12,  9,  8,  7,  2,  0,  1,  4,  5],
        [4, 10,  8,  7,  6,  2,  1,  0,  4,  4],
        [3, 10,  9,  5,  6,  2,  4,  4,  0,  3],
        [0,  7,  5,  3,  3,  4,  5,  4,  3,  0]
        ])

    #Travel time matrix
    timeCost = array([
        [0,  6,  6,  4,  4,  5,  6,  5,  4,  0],
        [6,  0,  4,  6,  5, 12, 13, 11, 11,  6],
        [6,  4,  0,  6,  3, 10, 10,  9, 10,  6],
        [4,  6,  6,  0,  5,  7,  9,  8,  6,  4],
        [4,  5,  3,  5,  0,  7,  8,  7,  7,  4],
        [5, 12, 10,  7,  7,  0,  3,  3,  3,  5],
        [6, 13, 10,  9,  8,  3,  0,  2,  5,  6],
        [5, 11,  9,  8,  7,  3,  2,  0,  5,  5],
        [4, 11, 10,  6,  7,  3,  5,  5,  0,  4],
        [0,  6,  6,  4,  4,  5,  6,  5,  4,  0]
        ])

    #The initial routes for Task 4
    #Each row describe the customers visited in a route. If the n'th index in a row is '1.0', then the route visits customer n.
    routes = array([
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1]
    ])

    #The distance cost of the initial routes.
    costRoutes = array([15.0, 12.0, 22.0, 18.0, 15.0, 22.0, 18.0, 10.0, 15.0, 11.0, 13.0, 12.0])

    #For Task 5. Input the routes you found in Task 2
    routes2 = array([
                     [0, 0, 0, 1, 0, 1, 1, 0],
                     [0, 0, 0, 0, 1, 1, 0, 1],
                     [1, 0, 1, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 1, 1],
                     [0, 1, 1, 1, 0, 0, 0, 0],
                 ])
    costRoutes2 = array([15.0, 12.0, 15.0, 22.0, 10.0, 11.0, 13.0])


    nodes = [(3.5,3.5),(0,0),(0,2),(3.5,1.5),(1,2.5),(5,5),(4,6.5),(3.5,5.5),(6,4)]
    labels = list(range(9))

    @staticmethod
    def plot_points(outputfile_name=None):
        "Plot instance points."
        style='bo'
        plt.plot([node[0] for node in data.nodes], [node[1] for node in data.nodes], style)
        plt.plot([data.nodes[0][0]], [data.nodes[0][1]], "rs")
        for (p, node) in enumerate(data.nodes):
            plt.text(node[0], node[1], '  '+str(data.labels[p]))
        plt.axis('scaled'); plt.axis('off')
        if outputfile_name is None:
            plt.show()
        else:
            plt.savefig(outputfile_name)

    @staticmethod
    def plot_routes(points, style='bo-'):
        "Plot lines to connect a series of points."
        for route in points:
            plt.plot(list(map(lambda p: data.nodes[p][0], route)), list(map(lambda p: data.nodes[p][1], route)), style)
        data.plot_points()


#if __name__ == "__main__":
#    data.plot_points()
#    data.plot_routes([[0,1,2,3,0],[0,4,0],[0,7,6,5,8,0]])
import logging
from itertools import chain, combinations, permutations
#logging.getLogger('pyomo.core').setLevel(logging.ERROR)


###### Utility ######
def binary_route_to_idx(route):
    return map(lambda x: x[0], filter(lambda x: x[1] == 1, enumerate(route)))

def check_route_feasibility(route, data, log=False):
    l = permutations(binary_route_to_idx(route))

    for perm in l:
        accum = 0
        prev = 0
        invalidRoute = False
        for c in perm:
            travelTime = accum + data.timeCost[prev][c]
            accum = max(travelTime, data.twStart[c])

            if accum > data.twEnd[c]:
                if log:
                    print("invalid route", perm, c, accum)
                invalidRoute = True
                break
            prev = c
        if not invalidRoute:
            if log:
                print("valid route", perm)
            return True
    return False

import pyomo.environ as po
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

###### MP/RMP/Relaxed (task 1,3,6) ######
def model(omega, data, c, domain=po.Binary, tee=False, ls=None, withDuals=False, SRCuts=None):
    R = list(range(len(omega)))

    m = po.ConcreteModel("model")

    def a(i, r):
        return omega[r][i]

    N = list(range(data.cust))

    def b(i,j,k,r):
        if a(i, r) + a(j, r) + a(k, r) >= 2:
            return 1
        else:
            return 0

    theta_domain = R
    m.theta = po.Var(theta_domain, domain=domain)

    if withDuals:
        m.dual = po.Suffix(direction=po.Suffix.IMPORT)

    if SRCuts is not None:
        model.cut_domain = po.Set(initialize=SRCuts)

        def makeSrCut(model, i,j,k):
            #i,j,k = x
            return 1 >= sum(b(i,j,k,r) * m.theta[r] for r in R)
        m.sr_cuts = po.Constraint(model.cut_domain, rule=makeSrCut)

    additional = 0

    m.activated_routes_delivery_cars = po.ConstraintList()

    def makeActivationC(model, i):
        return 1 == sum(
                a(i, r) * m.theta[r]
            for r in R)

    m.N = po.RangeSet(0, data.cust - 1)

    if ls is not None:
        #additional = sum(ls * (sum(a(i, r) * m.theta[r] for r in R) - 1) for i in N)
        additional = ls * (sum(m.theta[r] for r in R) - data.m)
    else:
        print("Running without lambdas")
        m.activated_routes_delivery_cars.add(data.m >= sum(m.theta[r] for r in R))

    m.activation_constraint = po.Constraint(m.N, rule=makeActivationC)

    m.value = po.Objective(expr=sum(int(c[r]) * m.theta[r] for r in R) + additional, sense=po.minimize)

    if tee:
        print("Solving...")

    now = time.time()

    res = po.SolverFactory("gurobi").solve(m, tee=(tee))

    later = time.time()
    difference = int(later - now)

    if tee:
        print("Solved in", difference, "seconds")

    if SRCuts is not None:
        return m, res, b
    else:
        return m, res

###### Running relaxed and IP (task 2) ######
import numpy as np

m, res = model(list(data.routes), data, domain=po.Binary, c=data.costRoutes)

print("OPTIMAL")
print("obj", m.value())
for r, _ in enumerate(list(data.routes)):
    if m.theta[r].value > 0:
        print(r, m.theta[r].value)
        print(data.routes[r])

m2, res2 = model(list(data.routes), data, domain=po.NonNegativeReals, c=data.costRoutes)

print("LP RELAXED")
print("obj", m2.value())
for r, _ in enumerate(list(data.routes)):
    if m2.theta[r].value > 0:
        print(r, m2.theta[r].value)
        #print(data.routes[r])

###### Lagrangian relaxed (task 2) ######
# Stole this code from exercise
def heuristic_sol(x, omega, cL):
    R = list(range(len(omega)))
    x_h=np.copy(x)
    # Find the un-covered customers
    uncovered = set(range(data.cust))
    for r, a in enumerate(x_h):
        if a == 1:
            for cidx, c in enumerate(omega[r]):
                if c == 1:
                    uncovered.discard(cidx)

    for c in uncovered:
        for ri, r in enumerate(omega):
            if r[c] == 1:
                x_h[ri]=1
                break

    z_UB = sum(int(cL[r]) * x_h[r] for r in R)
    return z_UB

def solve_lagrangian_dual(omega, data, mu=2, iterations=10):
    zs_LR=[]
    zs_LB=[]
    zs_UB=[]
    z_best_LB=-1000
    lambdas_ = []
    lambdas = np.array([0])
    for t in range(iterations):
        print(f"Lambdas {lambdas}")
        mL, resL = model(list(omega), data, domain=po.Binary, ls=lambdas[0], c=data.costRoutes)
        z_LR = mL.value()
        x = [v() for v in mL.component_data_objects(po.Var, active=True)]
        if z_best_LB<z_LR:
            z_best_LB=z_LR
        z_UB = heuristic_sol(x, omega, data.costRoutes)
        gamma = np.array([data.m - np.transpose(np.array(omega)) @ x])[0]
        theta = mu * ( z_UB - z_LR )/(sum(gamma**2))
        for i in range(len(lambdas)):
            if (gamma[i]>0.1):
                lambdas[i] = max(lambdas[i] + theta * gamma[i], 0)
        zs_UB.append(z_UB); zs_LR.append(z_LR); zs_LB.append(z_best_LB); lambdas_.append(lambdas)
    return lambdas

###### Utility ######
def tt(o):
    return (o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7])

def get_column_gen_lambdas(m, log=False, mu_constraint_ids=None):
    lambda1s = []
    lambda0 = 0
    mus = {}
    for c in m.component_objects(po.Constraint, active=True):
        if c.getname() == "activation_constraint":
            for index in c:
                lambda1s.append(m.dual[c[index]])
        elif c.getname() == "sr_cuts":
            for i,j,k in c:
                mus[(i,j,k)] = m.dual[c[(i,j,k)]]
        else:
            lambda0 = m.dual[c[1]]
        if log:
            print ("Constraint",c.getname())
        if log:
            for index in c:
                print (f"λ_{index}:", m.dual[c[index]])


    if mu_constraint_ids is not None:
        if log:
            print("mus")
            for k, v in mus:
                print(k, "->", v)
        return lambda1s, lambda0, mus
    else:
        return lambda1s, lambda0

###### Pricing problem/with optional SR-cuts (task 3) ######
def shortest_path_price(data, lambda1s, lambda0, mus=None):
    customers = range(data.cust + 1)
    V = customers
    N = list(filter(lambda x: x != 0, V))

    # Generate all customer 2-permutations with 0
    A = list(permutations(V, 2))

    bigM = 232183728

    m = po.ConcreteModel("price")
    m.x = po.Var(A, domain=po.Binary)
    m.alpha = po.Var(A, domain=po.NonNegativeReals)
    m.t = po.Var(V, domain=po.NonNegativeReals)
    m.y = po.Var(V, domain=po.NonNegativeReals)

    additional = 0
    if mus is not None:
        m.z = po.Var(mus.keys(), domain=po.Binary)
        additional = sum(mus[(i,j,k)] * m.z[(i,j,k)] for i,j,k in mus.keys())
        m.mus_constraint = po.ConstraintList()
        for i,j,k in mus.keys():
            def to_s(vertexId):
                return list(filter(lambda x: x[0] == vertexId, A))

            m.mus_constraint.add(
                sum(m.x[(i2, s)] for i2,s in to_s(i)) +
                sum(m.x[(j2, s)] for j2,s in to_s(j)) +
                sum(m.x[(k2, s)] for k2,s in to_s(k)) - 1 <= 2 * m.z[(i,j,k)]
            )

    first = sum((data.cost[i][j] - lambda1s[i - 1])*m.x[(i,j)] for i,j in A)
    m.value = po.Objective(expr= first - lambda0 - additional, sense=po.minimize)

    m.flow_balance = po.ConstraintList()
    for k in N:
        sinkK = list(filter(lambda ij: ij[1] == k, A))
        sourceK = list(filter(lambda ij: ij[0] == k, A))
        m.flow_balance.add(sum(m.x[(i,k)] for i,j in sinkK) == sum(m.x[(k,j)] for i,j in sourceK))

    m.from_zero_flow = po.ConstraintList()
    sinkZero = list(filter(lambda ij: ij[1] == 0, A))
    sourceZero = list(filter(lambda ij: ij[0] == 0, A))
    m.from_zero_flow.add(1 == sum(m.x[(i,0)] for i,j in sinkZero))
    m.from_zero_flow.add(1 == sum(m.x[(0,j)] for i,j in sourceZero))

    m.alpha_max = po.ConstraintList()
    for i,j in A:
        if j != 0:
            m.alpha_max.add(m.alpha[(i,j)] >= data.twStart[j - 1]) # twStart is -1 indexed relative to customer list
            m.alpha_max.add(m.alpha[(i,j)] >= data.timeCost[(i,j)] + m.t[i])

    m.time_constraint = po.ConstraintList()
    for i,j in A:
        if j != 0:
            m.time_constraint.add(m.alpha[(i,j)] + (bigM * m.x[(i,j)]) <= m.t[j] + bigM)

    m.arrival_constraint = po.ConstraintList()
    for i in V:
        if i == 0:
            m.arrival_constraint.add(m.t[i] == 0)
        else:
            m.arrival_constraint.add(0 <= m.t[i])
            m.arrival_constraint.add(m.t[i] <= data.twEnd[i - 1]) # twStart is -1 indexed relative to customer list

    m.capacity_constraint = po.ConstraintList()
    for i,j in A:
        if j != 0:
            m.capacity_constraint.add(m.y[i] + data.demand[j - 1] + (bigM * m.x[(i,j)]) <= m.y[j] + bigM)

    m.capacity_range = po.ConstraintList()
    for i in V:
        m.capacity_range.add(0 <= m.y[i])
        m.capacity_range.add(m.y[i] <= data.Q)

    res = po.SolverFactory("gurobi").solve(m, tee=(False))

    return m, res

###### Get reduced cost and post-process variables to paths... ######
def get_reduced_cost(m, data, log=False, mus=None, lambdas=None):
    if lambdas is None:
        lambda1s, lambda0 = get_column_gen_lambdas(m, log=log)
    else:
        lambda1s, lambda0 = lambdas

    ms, rs = shortest_path_price(data, lambda1s, lambda0, mus)

    allItems = range(data.cust + 1)
    i = 0
    path = [0]
    cost = 0

    done = False
    while not done:
        for j in allItems:
            if i != j and ms.x[(i, j)].value != 0:
                path.append(j)
                cost = cost + data.cost[i][j]
                if j == 0:
                    done = True
                    break
                i = j

    if log:
        print("path", path)

    pSet = set(map(lambda x: x - 1, filter(lambda x: x != 0, path)))

    def gen_01():
        for i in range(data.cust):
            if i in pSet:
                yield 1
            else:
                yield 0

    return ms.value(), list(gen_01()), path, cost

###### "Column gen by hand", latex table generator (task 4) ######
def column_gen_simple(initial, costF, remaining, data, thetas=False, violation=False, colTable=False, log=False):
    pointsInBasis = initial

    m, res = model(initial, data, domain=po.NonNegativeReals, withDuals=True, c = costF)

    lambda1s, lambda0 = get_column_gen_lambdas(m)

    objV, new_route, prettyPath, newCost = get_reduced_cost(m, data, log=log)

    print(new_route)

    print(f"r = {prettyPath}\\\\")
    print(f"c_r = {newCost}\\\\")

    print("\\hat{c} = " + str(objV) + "\\\\")

    if thetas:
        for idx, _ in enumerate(pointsInBasis):
            asStr = "%.2f" % m.theta[idx].value
            print("\\theta_{" + str(idx) + "}" + f" = {asStr}\\\\")

    objv = m.value()
    obS = "%.2f" % objv
    print(f"cost = {obS}\\\\")

    print("\\lambda_{1,r} = (", end='')
    print((",").join(list(map(lambda x: "%.2f" % x, lambda1s))), end='')
    print(")\\\\")

    print(f"\\lambda_0 = {lambda0}")
    print()

    combined = initial + remaining

    remainingSet = set(remaining)

    if colTable:
        cs = len(initial) * "c|"

        beginS = """\\[
            \\begin{tabular}{ |c|""" + cs +"""c| }
                \\hline"""
        endS = """\\hline
            \\end{tabular}
            \\]"""

        costs = ""
        for i in range(len(pointsInBasis)):
            costs = costs + str(int(costF[i])) + " & "

        header = "\\backslashbox{$\\lambda$}{$c_r$} & " + costs + " \\\\\n\\hline"

        body = ""
        for idx, l in enumerate(lambda1s):
            row = f"${l}$ & "
            for route in pointsInBasis:
                row = row + str(route[idx]) + " & "
            row = row + " = 1\\\\"
            body = body + row + "\n"

        lastRow = f"${lambda0}$ & "
        for _ in pointsInBasis:
            lastRow = lastRow + "1" + " & "
        lastRow = lastRow + "$\\leq 3$\\\\\n\\hline"

        thetasRow = "$\\theta$ & "
        for i in range(len(pointsInBasis)):
            thetasRow = thetasRow + str(m.theta[i].value) + " & "
        thetasRow = thetasRow + "\\\\"

        print(beginS)
        print(header)
        print(body, end='')
        print(lastRow)
        print(thetasRow)
        print(endS)

    if violation:
        combined = sorted(combined, key=lambda x: x[0])
        violated = 0
        for ridx, route in enumerate(combined):
            print("$\\theta_{" + str(ridx) + "}$ &", end='')
            for cust, r in enumerate(route):
                if r == 1:
                    print(("%.2f" % lambda1s[cust]) + " &", end='')
                else:
                    print("& ", end='')
            print(lambda0, end='')
            print(f"& $\leq {data.costRoutes[ridx]}$ \\\\")
        print(f"&& & & & &  && & {lambda0} & $\leq 0.0$ \\\\")

        print()
        print("violated", violated)

    return m, res, objv

###### "Column gen by hand", invocation and parameters (task 4) ######
tple = list(map(lambda o: tt(o), data.routes))
withCosts= list(zip(data.costRoutes, tple))

initialSubset = [withCosts[3], withCosts[4], withCosts[11]]
initialSubset = initialSubset + [(15, (1, 1, 0, 0, 0, 0, 0, 0))]
initialSubset = initialSubset + [(20, (0, 0, 1, 1, 0, 1, 0, 0))]
initialSubset = initialSubset + [(23, (1, 0, 0, 0, 1, 0, 1, 0))]
initialSubset = initialSubset + [(15, (0, 0, 1, 1, 0, 0, 0, 1))]

initialSubset = initialSubset + [(14, (1, 0, 0, 1, 0, 0, 0, 0))]
initialSubset = initialSubset + [(17, (0, 1, 0, 1, 0, 0, 1, 0))]
initialSubset = initialSubset + [(13, (0, 0, 1, 0, 1, 0, 0, 0))]
initialSubset = initialSubset + [(13, (0, 0, 0, 0, 1, 1, 0, 1))]
initialSubset = initialSubset + [(15, (1, 0, 1, 1, 0, 0, 0, 0))]

m, res, objv = column_gen_simple(list(map(lambda x: x[1], initialSubset)), list(map(lambda x: x[0], initialSubset)), [], data, colTable=False, log=False, thetas=True, violation=False)

###### Iterative column gen with pricing, includes optional SR-cuts (task 5 and 6) ######
def find_violated(m, data, omegaHat, b):
    threeCombinations = list(combinations(range(data.cust), 3))

    newConstraints = []
    for i,j,k in threeCombinations:
        constraintValue = sum(b(i,j,k,r) * m.theta[r].value for r in range(len(omegaHat)))
        if constraintValue > (1 + 0.01):
            newConstraints.append((i,j,k))
        else:
            pass

    return newConstraints

def column_generation_RMP(omegaHatWithCost, data, SRCuts=False):
    it = 0
    c = list(map(lambda x: x[0], omegaHatWithCost))
    omegaHat = list(map(lambda x: x[1], omegaHatWithCost))
    SRConstraints = []
    forceSrCut = False

    while True:
        print(f"#################### Iteration {it} ####################")
        if SRCuts:
            m, res, b = model(omegaHat, data, domain=po.NonNegativeReals, withDuals=True, c=c, SRCuts=SRConstraints)

            for idx in range(len(omegaHat)):
                if m.theta[idx].value != 0:
                    print("theta_", idx, " = ", m.theta[idx].value)

            lambda1s, lambda0, mus = get_column_gen_lambdas(m, mu_constraint_ids=SRConstraints)
            objV, path, prettyPath, newCost = get_reduced_cost(m, data, log=True, mus=mus, lambdas=(lambda1s, lambda0))
        else:
            m, res = model(omegaHat, data, domain=po.NonNegativeReals, withDuals=True, c=c)
            lambda1s, lambda0 = get_column_gen_lambdas(m)
            objV, path, prettyPath, newCost = get_reduced_cost(m, data, log=True)

        if tt(path) in set(map(lambda o: tt(o), omegaHat)):
            print("duplicate", path)
            if SRCuts and not forceSrCut:
                forceSrCut = True
            else:
                return m, res, omegaHat, c, SRConstraints

        if (SRCuts and objV >= 0) or forceSrCut:
            violated = find_violated(m, data, omegaHat, b)
            print("Violated", violated)

            if len(violated) == 0:
                return m, res, omegaHat, c, SRConstraints
            else:
                forceSrCut = False
                SRConstraints = SRConstraints + violated
        elif objV >= 0:
            return m, res, omegaHat, c, SRConstraints

        omegaHat.append(path)
        c.append(newCost)

        print(f"Reduced cost for this iteration was {objV} with cost {newCost} with route {prettyPath} or {path}")

        it = it + 1

###### Invocation of iterative column gen WITH SR-cuts (task 5 and 6) ######
#itemset = set([0, 1, 4, 5, 7, 9, 10])
#itemset = set([3,4,11])
itemset = range(len(data.routes))

omegaHatWithCost = list(map(lambda idx: (data.costRoutes[idx], list(data.routes[idx])), itemset))

m, r, optimalSet, c2, cuts = column_generation_RMP(omegaHatWithCost, data, SRCuts=True)

m2, r2 = model(optimalSet, data, domain=po.NonNegativeReals, withDuals=True, c=c2)
r2.write(num=1)
for t in range(len(optimalSet)):
    if m2.theta[t].value != 0:
        print(m2.theta[t].value, optimalSet[t])