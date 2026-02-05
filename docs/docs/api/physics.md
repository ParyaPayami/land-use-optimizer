# Physics Module API Reference

::: pimaluos.physics

---

## Traffic Simulation

### TrafficSimulator

Macroscopic traffic flow simulation using BPR function.

::: pimaluos.physics.engine.TrafficSimulator
    options:
      show_source: false
      members:
        - simulate
        - build_road_network_from_parcels
        - generate_trip_matrix
        - user_equilibrium_assignment

---

## Hydrology Simulation

### HydrologySimulator

Stormwater runoff modeling using Rational Method.

::: pimaluos.physics.engine.HydrologySimulator
    options:
      show_source: false
      members:
        - simulate
        - estimate_imperviousness
        - compute_runoff

---

## Solar Access

### SolarAccessSimulator

Shadow analysis and solar exposure modeling.

::: pimaluos.physics.engine.SolarAccessSimulator
    options:
      show_source: false
      members:
        - simulate
        - compute_shadow_impact
        - check_solar_access_violation

---

## Multi-Physics Engine

### MultiPhysicsEngine

Coordinated simulation across all physics domains.

::: pimaluos.physics.engine.MultiPhysicsEngine
    options:
      show_source: false
      members:
        - simulate_all
        - check_violations
        - compute_physics_penalty
        - prepare_scenario

---

## Digital Twin

### UrbanDigitalTwin

ML-Physics feedback loop for physics-informed optimization.

::: pimaluos.physics.digital_twin.UrbanDigitalTwin
    options:
      show_source: false
      members:
        - predict
        - validate
        - correct
        - run_feedback_loop

### DayNightCycle

Solar position calculations for shadow analysis.

::: pimaluos.physics.digital_twin.DayNightCycle
    options:
      show_source: false
      members:
        - get_sun_position
        - get_day_phase

### LODRenderer

Level-of-detail 3D building mesh generation.

::: pimaluos.physics.digital_twin.LODRenderer
    options:
      show_source: false
      members:
        - render_parcels
        - generate_building_mesh
