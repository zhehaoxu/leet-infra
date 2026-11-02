import json
import solution as s

CITY_STATE = {"A": "S1", "B": "S1", "C": "S2"}
QUERY = {"org": "A", "dest": "S1", "days": 3, "visiting_city_number": 1,
         "people_number": 2, "budget": 1500}


def base():
    return [
        {"current_city": "from A to B",
         "transportation": "Self-driving, from A to B, time: 1.5 hours",
         "breakfast": "R1, A", "lunch": "R2, A", "dinner": "R3, A",
         "attraction": "X, A;", "accommodation": "H2, B"},
        {"current_city": "B", "transportation": "-",
         "breakfast": "R4, B", "lunch": "R5, B", "dinner": "R6, B",
         "attraction": "Y, B;", "accommodation": "H2, B"},
        {"current_city": "from B to A",
         "transportation": "Self-driving, from B to A, time: 1.5 hours",
         "breakfast": "R7, B", "lunch": "R8, B", "dinner": "R9, B",
         "attraction": "Z, B;", "accommodation": "-"},
    ]


SANDBOX = {
    "restaurants": [{"name": "R%d" % i, "city": c} for c in "AB" for i in range(1, 10)],
    "accommodations": [{"name": "H2", "city": "B", "minimum_nights": 2}],
    "attractions": [{"name": x, "city": c} for x, c in (("X", "A"), ("Y", "B"), ("Z", "B"))],
    "flights": {"F1": {"origin": "B", "dest": "A"}},
    "distances": {("A", "B", "self-driving"): 100.0, ("B", "A", "self-driving"): 100.0},
}


def show(tag, itin, query=None, sandbox=None, csm=None):
    out = s.solve(query or dict(QUERY), itin, csm or CITY_STATE, sandbox or SANDBOX)
    print(tag, "passed=", out["passed"], "failed=", out["failed"])
    for name in out["failed"]:
        print("   ", name, "->", out["checks"][name]["msg"])


show("BASE", base())

it = base()
it[0]["dinner"] = "R1, A"
show("DUP-REST", it)

it = base()
it[1]["attraction"] = "Y, B;Y, B;"
show("DUP-ATTR", it)

it = base()
it[2]["transportation"] = "Flight Number: F1, from B to A, time: 1 hours"
show("MIXED-TRANS", it)

it = base()
it[2]["transportation"] = "Flight Number: F9, from B to A"
show("BAD-FLIGHT", it)

it = base()
it[0]["transportation"] = "Taxi, from A to B"
it[2]["transportation"] = "Taxi, from B to A"
show("TAXI-NODIST", it)

it = base()
it[2]["current_city"] = "from B to C"
show("OPEN-CIRCLE", it)

it = base()
it[2]["current_city"] = "B"
show("NOT-CLOSED", it)

print("--- 城市序列直测")
for seq in (["A", "B", "A"], ["A", "B", "B", "A"], ["A", "B", "C", "A"],
            ["A", "A", "B", "B", "A"], ["A", "B", "A", "B", "A"], ["A", "B"], ["A", "B", "C", "B", "A"]):
    print(seq, s._valid_city_sequence(seq))
