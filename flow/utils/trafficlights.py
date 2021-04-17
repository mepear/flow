def get_phase(name, ts):
    phases = all_phases[name]
    for i, phase in enumerate(phases):
        phase['duration'] = phase['minDur'] = phase['maxDur'] = str(ts[i % len(ts)])
    return phases

all_phases = {}

all_phases['center'] = [{
    "state": "GGggrrrrGGggrrrr"
}, {
    "state": "yyyyrrrryyyyrrrr"
}, {
    "state": "rrrrGGggrrrrGGgg"
}, {
    "state": "rrrryyyyrrrryyyy"
}]

all_phases['bottom'] = [{
    "state": "rrrGGgGgg",
}, {
    "state": "rrryyyyyy",
}, {
    "state": "GGgGrrrrr",
}, {
    "state": "yyyyrrrrr",
}]

all_phases['top'] = [{
    "state": "GggrrrGGg",
}, {
    "state": "yyyrrryyy",
}, {
    "state": "rrrGGgGrr",
}, {
    "state": "rrryyyyrr",
}]

all_phases['right'] = [{
    "state": "GGgGggrrr",
}, {
    "state": "yyyyyyrrr",
}, {
    "state": "GrrrrrGGg",
}, {
    "state": "yrrrrryyy",
}]

all_phases['left'] = [{
    "state": "GggrrrGGg",
}, {
    "state": "yyyrrryyy",
}, {
    "state": "rrrGGgGrr",
}, {
    "state": "rrryyyyrr",
}]

all_phases['left_in'] = [{
    "state": "GggrrrGGgrr",
}, {
    "state": "yyyrrryyyrr",
}, {
    "state": "rrrGGgGrrrr",
}, {
    "state": "rrryyyyrrrr",
}, {
    "state": "rrgGrrrrrGG",
}, {
    "state": "rryyrrrrryy"
}]

all_phases['top_in'] = [{
    "state": "rrrrrGGgGrr",
}, {
    "state": "rrrrryyyyrr",
}, {
    "state": "rrGggrrrGGg",
}, {
    "state": "rryyyrrryyy",
}, {
    "state": "GGrrrrrrrrG",
}, {
    "state": "yyrrrrrrrry"
}]

