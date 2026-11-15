"""A delivery from engaging to the depot, across the sea and back to the driver."""

from __future__ import annotations

from core.mission import CROSSED_M, DriveEnd, Mission, Observation, State

JOB = ("stokes", "plymouth")
CALAIS = (-30790.0, -3460.0)


def seen(
    beacon: bool = False, job=JOB, paused: bool = False, at=(0.0, 0.0), nav: float = 400_000.0
) -> Observation:
    return Observation(paused=paused, job=job, beacon=beacon, position=at, nav_distance=nav)


def test_without_a_job_nothing_is_driven() -> None:
    mission = Mission()
    assert not mission.observe(seen(beacon=True, job=None, nav=0.0))
    assert mission.state is State.NO_JOB


def test_the_beacon_switched_on_starts_a_drive() -> None:
    mission = Mission()
    assert not mission.observe(seen())
    assert mission.state is State.READY
    assert mission.observe(seen(beacon=True))
    assert mission.state is State.DRIVING
    assert not mission.observe(seen(beacon=True)), "one drive at a time"


def test_a_beacon_already_on_when_it_starts_must_go_off_first() -> None:
    mission = Mission()
    assert not mission.observe(seen(beacon=True))
    assert not mission.observe(seen(beacon=True))
    mission.observe(seen(beacon=False))
    assert mission.observe(seen(beacon=True))


def test_paused_the_beacon_is_not_heard() -> None:
    mission = Mission()
    mission.observe(seen())
    assert not mission.observe(seen(beacon=True, paused=True))
    assert mission.observe(seen(beacon=True)), "heard once the game runs again"


def test_across_the_sea_it_waits_for_the_crossing_and_the_beacon_again() -> None:
    """Calais: the drive ends at the train; the driver takes it, drives off the
    platform, and switches the beacon off and on."""
    mission = Mission()
    mission.observe(seen())
    mission.observe(seen(beacon=True))
    mission.drive_ended(DriveEnd.AT_CROSSING, CALAIS)
    assert mission.state is State.CROSSING
    assert not mission.observe(seen(beacon=True, at=CALAIS)), "still on this side"
    assert not mission.observe(seen(beacon=False, at=CALAIS))
    assert not mission.observe(seen(beacon=True, at=CALAIS)), "not across yet"
    folkestone = (CALAIS[0] - CROSSED_M - 300.0, CALAIS[1] - 3000.0)
    assert not mission.observe(seen(beacon=True, at=folkestone)), "armed at Calais, spent there"
    assert mission.state is State.READY
    mission.observe(seen(beacon=False, at=folkestone))
    assert mission.observe(seen(beacon=True, at=folkestone))


def test_at_the_depot_it_waits_for_the_next_job() -> None:
    mission = Mission()
    mission.observe(seen())
    mission.observe(seen(beacon=True))
    mission.drive_ended(DriveEnd.AT_DEPOT, (0.0, 0.0))
    assert mission.state is State.ARRIVED
    assert not mission.observe(seen(beacon=True)), "a beacon left on does nothing"
    mission.observe(seen(job=None, nav=0.0))
    assert mission.state is State.NO_JOB
    mission.observe(seen(job=("tesco", "bristol")))
    assert mission.state is State.READY


def test_a_safety_stop_is_driven_on_from_only_when_engaged_again() -> None:
    mission = Mission()
    mission.observe(seen())
    mission.observe(seen(beacon=True))
    mission.drive_ended(DriveEnd.UNSAFE, (0.0, 0.0), "-6.9 m from the path")
    assert mission.state is State.STOPPED
    assert "-6.9 m from the path" in mission.message
    assert not mission.observe(seen(beacon=True))
    mission.observe(seen(beacon=False))
    assert mission.observe(seen(beacon=True))


def test_switched_off_by_the_driver_it_is_ready_again() -> None:
    mission = Mission()
    mission.observe(seen())
    mission.observe(seen(beacon=True))
    mission.drive_ended(DriveEnd.DISENGAGED, (0.0, 0.0))
    assert mission.state is State.READY
    assert mission.observe(seen(beacon=False)) is False
    assert mission.observe(seen(beacon=True))


def test_a_route_s_end_is_the_depot_or_a_crossing_by_where_the_truck_is() -> None:
    from core.mission import how_it_ended

    assert how_it_ended("end_of_route", "", 86.0, 40_000.0)[0] is DriveEnd.AT_DEPOT
    assert how_it_ended("end_of_route", "", 541_900.0, 120.0)[0] is DriveEnd.AT_CROSSING
    short = how_it_ended("end_of_route", "", 300_000.0, 9_000.0)
    assert short[0] is DriveEnd.UNSAFE and "short" in short[1]


def test_how_a_drive_ended_as_follow_py_tells() -> None:
    from core.mission import how_it_ended

    assert how_it_ended("disengaged", "", 1e5, 1e5)[0] is DriveEnd.DISENGAGED
    assert how_it_ended("unsafe", "-6.0 m from the path", 1e5, 1e5) == (
        DriveEnd.UNSAFE,
        "-6.0 m from the path",
    )
    assert how_it_ended("lost_game", "gone", 1e5, 1e5)[0] is DriveEnd.LOST_GAME
    assert how_it_ended(None, "", 1e5, 1e5) == (DriveEnd.UNSAFE, "the drive did not start")


def test_a_job_taken_while_paused_is_seen() -> None:
    """The job is chosen in the game's menus, which pause it."""
    mission = Mission()
    assert not mission.observe(seen(paused=True))
    assert mission.state is State.READY


def test_engaged_at_the_depot_it_does_not_drive_into_the_yard() -> None:
    """Plymouth, the job not yet delivered: 0 m to go."""
    mission = Mission()
    mission.observe(seen(nav=0.0))
    assert not mission.observe(seen(beacon=True, nav=0.0))
    assert mission.state is State.ARRIVED


def test_engaged_short_of_the_depot_it_drives_on() -> None:
    """Reloaded 738 m short of the depot in Milan, two lights still to come."""
    mission = Mission()
    mission.observe(seen(nav=738.0))
    assert mission.observe(seen(beacon=True, nav=738.0))


def test_at_the_trailer_hitching_it_makes_the_job_s_destination_ready() -> None:
    """Milan: driven to the depot the job's trailer waited at, then on to
    the destination once it was hitched."""
    trailer_at, destination = ("ika_bohag", "milano"), ("cargotras", "milano")
    mission = Mission()
    mission.observe(seen(job=trailer_at))
    assert mission.observe(seen(beacon=True, job=trailer_at))
    mission.drive_ended(DriveEnd.AT_DEPOT, (0.0, 0.0))
    assert mission.state is State.ARRIVED
    mission.observe(seen(job=destination))
    assert mission.state is State.READY
    assert mission.observe(seen(beacon=True, job=destination))


def test_at_the_depot_the_delivery_is_the_driver_s() -> None:
    """The driver parks and delivers: the beacon there drives nothing, and
    the next job is ready."""
    mission = Mission()
    mission.observe(seen())
    mission.observe(seen(beacon=True))
    mission.drive_ended(DriveEnd.AT_DEPOT, (0.0, 0.0))
    assert mission.state is State.ARRIVED
    mission.observe(seen(beacon=False, nav=60.0))
    assert not mission.observe(seen(beacon=True, nav=60.0))
    mission.observe(seen(job=("tesco", "bristol")))
    assert mission.state is State.READY


def test_arrived_where_the_gps_went_a_place_marked_anew_is_driven_to() -> None:
    """Near Geneva, a place marked on the map reached and another marked:
    the job the same, the mission stood arrived at the first for good."""
    mission = Mission()
    mission.observe(seen(nav=50_000.0))
    mission.observe(seen(beacon=True, nav=50_000.0))
    mission.drive_ended(DriveEnd.AT_DEPOT, (0.0, 0.0))
    assert not mission.observe(seen(beacon=True, nav=80.0))
    assert mission.state is State.ARRIVED
    assert not mission.observe(seen(beacon=True, nav=12_000.0)), "the beacon must go off first"
    assert mission.state is State.READY
    mission.observe(seen(beacon=False, nav=12_000.0))
    assert mission.observe(seen(beacon=True, nav=12_000.0))


def test_a_route_on_the_gps_with_no_job_told_is_driven_to_and_stopped_at() -> None:
    """A job's trailer at its depot is not told until hitched (6 October)."""
    from core.mission import GPS_JOB

    mission = Mission()
    mission.observe(seen(job=None, nav=33_500.0))
    assert mission.state is State.READY and "GPS" in mission.message
    assert mission.observe(seen(beacon=True, job=None, nav=33_500.0))
    assert mission.to_gps
    mission.drive_ended(DriveEnd.AT_DEPOT, (0.0, 0.0))
    assert mission.state is State.ARRIVED
    mission.observe(seen(beacon=False, job=None, nav=50.0))
    assert not mission.observe(seen(beacon=True, job=None, nav=50.0)), "there: the driver's"
    # The trailer hitched by the driver: the job is told, and driven.
    mission.observe(seen(job=("cargotras", "bern"), nav=200_000.0))
    assert mission.state is State.READY and not mission.to_gps
    assert GPS_JOB != ("cargotras", "bern")


def test_no_job_and_no_route_is_no_job() -> None:
    mission = Mission()
    mission.observe(seen(job=None, nav=0.0))
    assert mission.state is State.NO_JOB


def test_a_job_s_place_far_from_where_the_game_s_gps_goes_is_not_driven_to() -> None:
    """Bremen: a job delivered, its places still told, the GPS set to a
    trailer 7.6 km off; the job's way was 1021 km."""
    from tools.plan_route import gps_elsewhere

    assert gps_elsewhere(7_600.0, 1_021_100.0)
    assert not gps_elsewhere(997_300.0, 1_008_200.0), "the job's: 11 km over 1000"
    assert not gps_elsewhere(12_000.0, 31_000.0), "within 30 km on a short way"
    assert gps_elsewhere(20_000.0, 60_000.0)
