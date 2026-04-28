from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureRegion:
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int

    @classmethod
    def centered_square(cls, *, width: int, height: int, size: int) -> "CaptureRegion":
        if size > width or size > height:
            raise ValueError(
                f"Centered square size ({size}) exceeds capture dimensions ({height}x{width})"
            )
        x_left = (width - size) // 2
        y_top = (height - size) // 2
        return cls(
            x_left=x_left,
            x_right=x_left + size,
            y_top=y_top,
            y_bottom=y_top + size,
        )

    def validate(self, *, width: int, height: int) -> None:
        if self.x_left < 0 or self.y_top < 0:
            raise ValueError(f"Capture region has negative bounds: {self}")
        if self.x_right > width or self.y_bottom > height:
            raise ValueError(
                f"Capture region {self} exceeds capture dimensions ({height}x{width})"
            )
        if self.x_left >= self.x_right or self.y_top >= self.y_bottom:
            raise ValueError(
                f"Capture region must have positive width and height: {self}"
            )
