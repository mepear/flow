import traci
traci.start(['sumo-gui', '-c', 'test.sumocfg', '--device.taxi.dispatch-algorithm', 'traci'])
step = 0
traci.route.add('r1', ['edge1', 'edge2'])
traci.route.add('r2', ['edge2', 'edge3'])
traci.route.add('r3', ['edge3', 'edge4'])
traci.route.add('r4', ['edge4', 'edge1'])

traci.person.add('per1', 'edge1', 0)
traci.person.appendDrivingStage('per1', 'edge2', 'taxi')
traci.person.setColor('per1', (255, 0, 0))
traci.vehicle.add('taxi1', 'r2', typeID='taxi')
while step < 1000:
    traci.simulationStep()
    step += 1
    reservations = traci.person.getTaxiReservations(False)
    print(traci.person.getStage('per1').description)
    if len(reservations) > 0:
        print(reservations)
        traci.vehicle.dispatchTaxi('taxi1', reservations[0].id)
traci.close()