"""
Geometrische Regel zur Erkennung einer Tordurchfahrt.

Ein Tor wird durch zwei Punkte definiert, linke und rechte Torstange,
oder allgemeiner durch eine gedachte Linie zwischen zwei Referenzpunkten.
Der Athlet wird durch einen einzelnen Bezugspunkt repraesentiert,
zum Beispiel die Fusspunktmitte der Bounding Box.

Eine Tordurchfahrt gilt als erkannt, wenn die Verbindungsstrecke
zwischen der Athletenposition im vorherigen und im aktuellen Frame
die Torlinie schneidet. Das wird ueber das Vorzeichen des
Kreuzprodukts bestimmt, ein reiner geometrischer Test ohne
Abhaengigkeit von Modell, Kamera oder Framerate.

Wichtig: Diese Regel liefert nur True oder False fuer "Linie geschnitten".
Ob der Schnittpunkt tatsaechlich zwischen den beiden Torstangen liegt,
prueft segments_intersect zusaetzlich. Nur wenn beides zutrifft,
gilt das Tor als passiert.
"""

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    """Achsenparallele Box mit expliziten Bildkoordinaten."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("BoundingBox muss eine positive Groesse haben")

    @classmethod
    def from_top_left_xywh(
        cls, x: float, y: float, width: float, height: float
    ) -> "BoundingBox":
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_center_xywh(
        cls, x: float, y: float, width: float, height: float
    ) -> "BoundingBox":
        return cls(x - width / 2, y - height / 2, x + width / 2, y + height / 2)

    @property
    def center(self) -> Point:
        return Point((self.left + self.right) / 2, (self.top + self.bottom) / 2)


@dataclass(frozen=True)
class BoxGateCrossing:
    """Boxbasierter Schnitt inklusive expliziter Unsicherheitsangabe."""

    fraction: float
    supporting_points: int
    uncertain: bool


def _cross(o: Point, a: Point, b: Point) -> float:
    """Kreuzprodukt der Vektoren OA und OB. Vorzeichen zeigt die Seite an."""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Prueft, ob die Strecke p1 bis p2 (Athletenbewegung)
    die Strecke p3 bis p4 (Torlinie) schneidet.
    Reiner geometrischer Test, keine Rundungstoleranz.
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def gate_crossed(
    prev_athlete_pos: Point,
    curr_athlete_pos: Point,
    gate_left: Point,
    gate_right: Point,
) -> bool:
    """
    Prueft, ob der Athlet zwischen zwei aufeinanderfolgenden Frames
    ein bestimmtes Tor passiert hat.

    prev_athlete_pos, curr_athlete_pos:
        Bezugspunkt des Athleten im vorherigen und aktuellen Frame,
        in Bildkoordinaten oder in einer entzerrten Weltkoordinate.
    gate_left, gate_right:
        Die beiden Referenzpunkte des Tores, zum Beispiel linke
        und rechte Torstange.
    """
    return segments_intersect(prev_athlete_pos, curr_athlete_pos, gate_left, gate_right)


def moving_gate_crossing_fraction(
    prev_athlete_pos: Point,
    curr_athlete_pos: Point,
    prev_gate_left: Point,
    prev_gate_right: Point,
    curr_gate_left: Point,
    curr_gate_right: Point,
    *,
    iterations: int = 40,
) -> float | None:
    """Bestimmt die relative Tordurchfahrt trotz bewegter Kamera.

    Athlet und beide Torpunkte werden zwischen zwei Frames linear
    interpoliert. Eine Rueckgabe von 0 bis 1 beschreibt den Zeitpunkt
    innerhalb des Frameintervalls. ``None`` bedeutet keine Durchfahrt.
    """

    def interpolate(start: Point, end: Point, fraction: float) -> Point:
        return Point(
            start.x + (end.x - start.x) * fraction,
            start.y + (end.y - start.y) * fraction,
        )

    def side(fraction: float) -> float:
        athlete = interpolate(prev_athlete_pos, curr_athlete_pos, fraction)
        left = interpolate(prev_gate_left, curr_gate_left, fraction)
        right = interpolate(prev_gate_right, curr_gate_right, fraction)
        return _cross(left, right, athlete)

    side_start = side(0.0)
    side_end = side(1.0)
    if side_start == 0 or side_end == 0 or side_start * side_end >= 0:
        return None

    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2.0
        if side_start * side(middle) <= 0:
            high = middle
        else:
            low = middle
    fraction = (low + high) / 2.0

    athlete = interpolate(prev_athlete_pos, curr_athlete_pos, fraction)
    left = interpolate(prev_gate_left, curr_gate_left, fraction)
    right = interpolate(prev_gate_right, curr_gate_right, fraction)
    gate_dx = right.x - left.x
    gate_dy = right.y - left.y
    gate_length_squared = gate_dx * gate_dx + gate_dy * gate_dy
    if gate_length_squared == 0:
        return None

    projection = (
        (athlete.x - left.x) * gate_dx + (athlete.y - left.y) * gate_dy
    ) / gate_length_squared
    return fraction if 0.0 <= projection <= 1.0 else None


def moving_box_gate_crossing(
    prev_athlete_box: BoundingBox,
    curr_athlete_box: BoundingBox,
    prev_gate_box: BoundingBox,
    curr_gate_box: BoundingBox,
    *,
    horizontal_samples: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> BoxGateCrossing | None:
    """Prueft die untere Athletenkante gegen die mitbewegte Torunterkante.

    Die Torunterkante approximiert die Linie zwischen den beiden Stangenfuessen.
    Mehrere Punkte entlang der unteren Athletenbox bilden Koerper und Ski robuster
    ab als ein einzelner Fusspunkt. Der Median der gueltigen Schnitte begrenzt den
    Einfluss einer einzelnen Boxkante.
    """

    if not horizontal_samples or any(
        sample < 0.0 or sample > 1.0 for sample in horizontal_samples
    ):
        raise ValueError("horizontal_samples muessen zwischen 0 und 1 liegen")

    prev_gate_left = Point(prev_gate_box.left, prev_gate_box.bottom)
    prev_gate_right = Point(prev_gate_box.right, prev_gate_box.bottom)
    curr_gate_left = Point(curr_gate_box.left, curr_gate_box.bottom)
    curr_gate_right = Point(curr_gate_box.right, curr_gate_box.bottom)

    fractions: list[float] = []
    for sample in horizontal_samples:
        prev_point = Point(
            prev_athlete_box.left
            + (prev_athlete_box.right - prev_athlete_box.left) * sample,
            prev_athlete_box.bottom,
        )
        curr_point = Point(
            curr_athlete_box.left
            + (curr_athlete_box.right - curr_athlete_box.left) * sample,
            curr_athlete_box.bottom,
        )
        fraction = moving_gate_crossing_fraction(
            prev_point,
            curr_point,
            prev_gate_left,
            prev_gate_right,
            curr_gate_left,
            curr_gate_right,
        )
        if fraction is not None:
            fractions.append(fraction)

    if not fractions:
        return None
    supporting_points = len(fractions)
    return BoxGateCrossing(
        fraction=float(median(fractions)),
        supporting_points=supporting_points,
        uncertain=supporting_points < 2,
    )
