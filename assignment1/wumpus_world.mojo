from std.random import random_float64, random_si64, seed


struct Percept(Copyable):
    var time_step: Int
    var bump: Bool
    var breeze: Bool
    var stench: Bool
    var scream: Bool
    var glitter: Bool
    var reward: Int
    var done: Bool

    def __init__(
        out self,
        time_step: Int,
        bump: Bool,
        breeze: Bool,
        stench: Bool,
        scream: Bool,
        glitter: Bool,
        reward: Int,
        done: Bool,
    ):
        self.time_step = time_step
        self.bump = bump
        self.breeze = breeze
        self.stench = stench
        self.scream = scream
        self.glitter = glitter
        self.reward = reward
        self.done = done

    def describe(self) -> String:
        return String(
            "time:",
            self.time_step,
            ": bump:",
            self.bump,
            ", breeze:",
            self.breeze,
            ", stench:",
            self.stench,
            ", scream:",
            self.scream,
            ", glitter:",
            self.glitter,
            ", reward:",
            self.reward,
            ", done:",
            self.done,
        )


struct Action:
    @staticmethod
    def left() -> Int:
        return 0

    @staticmethod
    def right() -> Int:
        return 1

    @staticmethod
    def forward() -> Int:
        return 2

    @staticmethod
    def grab() -> Int:
        return 3

    @staticmethod
    def shoot() -> Int:
        return 4

    @staticmethod
    def climb() -> Int:
        return 5

    @staticmethod
    def random_action() -> Int:
        return Int(random_si64(0, 5))

    @staticmethod
    def symbol(action: Int) -> String:
        if action == Action.left():
            return "TURN LEFT"
        if action == Action.right():
            return "TURN RIGHT"
        if action == Action.forward():
            return "FORWARD"
        if action == Action.grab():
            return "GRAB"
        if action == Action.shoot():
            return "SHOOT"
        if action == Action.climb():
            return "CLIMB"
        return "UNKNOWN"


struct Orientation:
    @staticmethod
    def east() -> Int:
        return 0

    @staticmethod
    def south() -> Int:
        return 1

    @staticmethod
    def west() -> Int:
        return 2

    @staticmethod
    def north() -> Int:
        return 3

    @staticmethod
    def symbol(orientation: Int) -> String:
        if orientation == Orientation.east():
            return ">"
        if orientation == Orientation.south():
            return "v"
        if orientation == Orientation.west():
            return "<"
        if orientation == Orientation.north():
            return "^"
        return "?"

    @staticmethod
    def turn_right(orientation: Int) -> Int:
        if orientation == Orientation.east():
            return Orientation.south()
        if orientation == Orientation.south():
            return Orientation.west()
        if orientation == Orientation.west():
            return Orientation.north()
        return Orientation.east()

    @staticmethod
    def turn_left(orientation: Int) -> Int:
        if orientation == Orientation.east():
            return Orientation.north()
        if orientation == Orientation.north():
            return Orientation.west()
        if orientation == Orientation.west():
            return Orientation.south()
        return Orientation.east()


struct Location(Copyable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int = 0, y: Int = 0):
        self.x = x
        self.y = y

    def is_same(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    def is_left_of(self, other: Self) -> Bool:
        return self.x < other.x and self.y == other.y

    def is_right_of(self, other: Self) -> Bool:
        return self.x > other.x and self.y == other.y

    def is_above(self, other: Self) -> Bool:
        return self.y > other.y and self.x == other.x

    def is_below(self, other: Self) -> Bool:
        return self.y < other.y and self.x == other.x

    def forward(mut self, orientation: Int, max_x: Int, max_y: Int) -> Bool:
        var bump = False
        if orientation == Orientation.west():
            if self.x == 0:
                bump = True
            else:
                self.x -= 1
        elif orientation == Orientation.east():
            if self.x == max_x:
                bump = True
            else:
                self.x += 1
        elif orientation == Orientation.north():
            if self.y == max_y:
                bump = True
            else:
                self.y += 1
        else:
            if self.y == 0:
                bump = True
            else:
                self.y -= 1
        return bump


struct Environment:
    var wumpus_location: Location
    var wumpus_alive: Bool
    var has_wumpus: Bool
    var agent_location: Location
    var agent_orientation: Int
    var agent_has_arrow: Bool
    var agent_has_gold: Bool
    var game_over: Bool
    var gold_location: Location
    var pit_indices: List[Int]
    var time_step: Int
    var pit_prob: Float64
    var allow_climb_without_gold: Bool
    var world_size_x: Int
    var world_size_y: Int

    def __init__(
        out self,
        world_size_x: Int = 4,
        world_size_y: Int = 4,
        agent_start_location_x: Int = 0,
        agent_start_location_y: Int = 0,
        agent_start_orientation: Int = Orientation.east(),
        agent_has_arrow: Bool = True,
        agent_has_gold: Bool = False,
        pit_prob: Float64 = 0.2,
        allow_climb_without_gold: Bool = False,
        has_wumpus: Bool = True,
    ):
        self.world_size_x = world_size_x
        self.world_size_y = world_size_y
        self.agent_location = Location(agent_start_location_x, agent_start_location_y)
        self.agent_orientation = agent_start_orientation
        self.agent_has_arrow = agent_has_arrow
        self.agent_has_gold = agent_has_gold
        self.pit_prob = pit_prob
        self.allow_climb_without_gold = allow_climb_without_gold
        self.has_wumpus = has_wumpus
        self.game_over = False
        self.time_step = 0
        self.pit_indices = List[Int]()
        var total_cells = world_size_x * world_size_y
        var start_index = agent_start_location_y * world_size_x + agent_start_location_x
        var wumpus_index = start_index
        while wumpus_index == start_index:
            wumpus_index = Int(random_si64(0, Int64(total_cells - 1)))
        self.wumpus_location = Location(wumpus_index % world_size_x, wumpus_index // world_size_x)
        self.wumpus_alive = has_wumpus

        var gold_index = start_index
        while gold_index == start_index:
            gold_index = Int(random_si64(0, Int64(total_cells - 1)))
        self.gold_location = Location(gold_index % world_size_x, gold_index // world_size_x)

        for index in range(total_cells):
            if index == start_index:
                continue
            if random_float64(0.0, 1.0) < self.pit_prob:
                self.pit_indices.append(index)

    def total_cells(self) -> Int:
        return self.world_size_x * self.world_size_y

    def to_index(self, location: Location) -> Int:
        return location.y * self.world_size_x + location.x

    def from_index(self, index: Int) -> Location:
        return Location(index % self.world_size_x, index // self.world_size_x)

    def random_non_start_location(self) -> Location:
        var start_index = self.to_index(self.agent_location)
        while True:
            var index = Int(random_si64(0, Int64(self.total_cells() - 1)))
            if index != start_index:
                return self.from_index(index)

    def populate_pits(mut self):
        var start_index = self.to_index(self.agent_location)
        for index in range(self.total_cells()):
            if index == start_index:
                continue
            if random_float64(0.0, 1.0) < self.pit_prob:
                self.pit_indices.append(index)

    def get_percept(self) -> Percept:
        return Percept(self.time_step, False, self.is_breeze(), self.is_stench(), False, False, 0, False)

    def is_pit_at(self, location: Location) -> Bool:
        var wanted_index = self.to_index(location)
        for pit_index in self.pit_indices:
            if pit_index == wanted_index:
                return True
        return False

    def is_adjacent(self, first: Location, second: Location) -> Bool:
        var dx = first.x - second.x
        if dx < 0:
            dx = -dx
        var dy = first.y - second.y
        if dy < 0:
            dy = -dy
        return dx + dy == 1

    def is_pit_adjacent_to_agent(self) -> Bool:
        for pit_index in self.pit_indices:
            if self.is_adjacent(self.agent_location, self.from_index(pit_index)):
                return True
        return False

    def is_wumpus_adjacent_to_agent(self) -> Bool:
        return self.has_wumpus and self.is_adjacent(self.agent_location, self.wumpus_location)

    def is_agent_at_hazard(self) -> Bool:
        return self.is_pit_at(self.agent_location) or (self.is_wumpus_at(self.agent_location) and self.wumpus_alive)

    def is_wumpus_at(self, location: Location) -> Bool:
        return self.has_wumpus and self.wumpus_location.is_same(location)

    def is_agent_at(self, location: Location) -> Bool:
        return self.agent_location.is_same(location)

    def is_gold_at(self, location: Location) -> Bool:
        return self.gold_location.is_same(location)

    def is_glitter(self) -> Bool:
        return self.is_gold_at(self.agent_location)

    def is_breeze(self) -> Bool:
        return self.is_pit_adjacent_to_agent() or self.is_pit_at(self.agent_location)

    def is_stench(self) -> Bool:
        return self.is_wumpus_adjacent_to_agent() or self.is_wumpus_at(self.agent_location)

    def wumpus_in_line_of_fire(self) -> Bool:
        if self.agent_orientation == Orientation.east():
            return self.has_wumpus and self.agent_location.is_left_of(self.wumpus_location)
        if self.agent_orientation == Orientation.south():
            return self.has_wumpus and self.agent_location.is_above(self.wumpus_location)
        if self.agent_orientation == Orientation.west():
            return self.has_wumpus and self.agent_location.is_right_of(self.wumpus_location)
        return self.has_wumpus and self.agent_location.is_below(self.wumpus_location)

    def kill_attempt(mut self) -> Bool:
        if not (self.has_wumpus and self.wumpus_alive):
            return False
        var scream = self.wumpus_in_line_of_fire()
        self.wumpus_alive = not scream
        return scream

    def step(mut self, action: Int) -> Percept:
        var special_reward = 0
        var bump = False
        var scream = False
        var reward: Int

        if self.game_over:
            reward = 0
        else:
            if action == Action.left():
                self.agent_orientation = Orientation.turn_left(self.agent_orientation)
            elif action == Action.right():
                self.agent_orientation = Orientation.turn_right(self.agent_orientation)
            elif action == Action.forward():
                bump = self.agent_location.forward(self.agent_orientation, self.world_size_x - 1, self.world_size_y - 1)
                if self.agent_has_gold:
                    self.gold_location = Location(self.agent_location.x, self.agent_location.y)
                if self.is_agent_at_hazard():
                    special_reward = -1000
                    self.game_over = True
            elif action == Action.grab():
                if self.is_gold_at(self.agent_location):
                    self.agent_has_gold = True
            elif action == Action.shoot():
                if self.agent_has_arrow:
                    scream = self.kill_attempt()
                    special_reward = -10
                    self.agent_has_arrow = False
            elif action == Action.climb():
                if self.agent_location.is_same(Location(0, 0)):
                    if self.agent_has_gold:
                        special_reward = 1000
                    if self.allow_climb_without_gold or self.agent_has_gold:
                        self.game_over = True
            reward = -1 + special_reward

        var breeze = self.is_breeze()
        var stench = self.is_stench()
        var glitter = self.is_glitter()
        self.time_step += 1
        return Percept(self.time_step, bump, breeze, stench, scream, glitter, reward, self.game_over)

    def cell_string(self, location: Location) -> String:
        var a = " "
        if self.is_agent_at(location):
            a = Orientation.symbol(self.agent_orientation)
        var p = "P" if self.is_pit_at(location) else " "
        var w = " "
        if self.has_wumpus and self.is_wumpus_at(location):
            w = "W" if self.wumpus_alive else "w"
        var g = "G" if self.is_gold_at(location) else " "
        return String(a, p, w, g)

    def visualize(self):
        for y in range(self.world_size_y - 1, -1, -1):
            var line = String("|")
            for x in range(self.world_size_x):
                line += self.cell_string(Location(x, y))
                line += "|"
            print(line)


struct NaiveAgent:
    def __init__(out self):
        pass

    def choose_action(self) -> Int:
        return Action.random_action()

    def run(self):
        seed()
        var env = Environment()
        var cumulative_reward = 0
        var percept = env.get_percept()

        while not percept.done:
            env.visualize()
            print("Percept:", percept.describe())
            var action = self.choose_action()
            print()
            print("Action:", Action.symbol(action))
            print()
            percept = env.step(action)
            cumulative_reward += percept.reward

        env.visualize()
        print("Percept:", percept.describe())
        print("Cumulative reward:", cumulative_reward)


def main():
    var agent = NaiveAgent()
    agent.run()