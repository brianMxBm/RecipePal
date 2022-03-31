import time
import bisect
import itertools


class Node:
    # Node Constructor Consists of cost, state and Steps/steps there.
    def __init__(self, cost, state, steps):
        self.cost = cost
        self.state = state
        self.steps = steps

    def __lt__(self, other):  # Function to check
        return self.cost < other.cost

    def printsteps(self):
        i = 0
        while i < len(self.steps):
            if (i + 1) % 3 == 0:
                print(self.steps[i], "<-")
                i += 1
            else:
                print(self.steps[i], self.steps[i + 1], "->")
                i += 2


class Tree:
    # Tree Constructor Consists of Initial state, states yet to be expanded on, cost, etc.
    def __init__(self, fitness):
        state = [0] * len(fitness)  # Initial state of zero's as the position
        steps = []
        cost = 0
        # Crate node object with cost zero, empty steps, and an initial state of all zeros
        root = Node(cost, state, steps)
        self.fringe = [root]
        self.fitness = fitness

    # Function to check if a state has already been visited as to not traverse it in the Tree.
    def stateVisited(self, node):
        for item in self.fringe:  # Traverse elements in state
            # If elements in the passed state(node) match the current state(node) and have a shorter overall steps cost
            if node.state == item.state and node.cost < item.cost:
                # Remove the visited state since a better steps was found
                self.fringe.remove(item)
                # Insert the state(node) passed using bisect (after exisitng entries)
                bisect.insort(self.fringe, node)
                return True
            # If elements in the passed state(node) match the current state(node)
            if node.state == item.state:
                return True
        return False  # Default to false

    # Function to generate staets "as we go", since this is a state space search problem it's better to do so.
    def stateCreation(self, node):
        fitness_len = len(self.fitness)
        if len(node.steps) % 3 == 0:
            for elem in itertools.combinations(list(range(fitness_len)), 2):
                if node.state[elem[0]] == 0 and node.state[elem[1]] == 0:
                    newstate = list(node.state)
                    newstate[elem[0]] = 1
                    newstate[elem[1]] = 1

                    newsteps = list(node.steps)
                    newsteps.extend([elem[0] + 1, elem[1] + 1])

                    newcost = node.cost + max(
                        self.fitness[elem[0]], self.fitness[elem[1]]
                    )

                    createdNode = Node(newcost, newstate, newsteps)

                    if not self.stateVisited(createdNode):
                        bisect.insort(self.fringe, createdNode)

                else:
                    for elem in list(range(fitness_len)):
                        if node.state[elem] == 1:
                            newstate = list(node.state)
                            newstate[elem] = 0

                            newsteps = list(node.steps)
                            newsteps.append(elem + 1)

                            newcost = node.cost + self.fitness[elem]

                            createdNode = Node(newcost, newstate, newsteps)

                        if not self.stateVisited(createdNode):
                            bisect.insort(self.fringe, createdNode)

    def stateSearch(self):
        visitedCounter = 0
        fitness_len = len(self.fitness)
        while True:
            if not self.fringe:
                return "NO SOLUTION FOUND"

            node = self.fringe.pop(0)
            visitedCounter += 1

            if node.state == [1] * fitness_len:
                print(node.cost, visitedCounter)
                node.printsteps()
                return "SOLUTION FOUND"

            self.stateCreation(node)


t = time.process_time()

# inp = [1,2,5,10] #(a)
inp = [1, 2, 5, 10, 3, 4, 14, 18, 20, 50]  # (b)
# inp = [1,2,5,10,12,17,24,21,20,20,11,33,15,19,55] #(c)
state_space = Tree(inp)
state_space.stateSearch()

elapsed_time = time.process_time() - t
