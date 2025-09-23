# manim -pql -i conv2d_forward.py

from manim import *

config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 30.0
config.frame_width = 50.0


class CustomScene(Scene):
    def construct(self):
        # fmt: off
        x = np.array([[1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1],
                      [0, 0, 1, 1, 0],
                      [0, 1, 1, 0, 0]])
        w = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [1, 0, 1]])
        # fmt: on

        d = 6

        matrix_5x5_input = (
            VGroup(*[Square().set_fill(DARK_BLUE, opacity=1.0) for i in range(5) for j in range(5)])
            .arrange_in_grid(5, 5, buff=0.0)
            .move_to(LEFT * d)
        )
        matrix_3x3_filter = (
            VGroup(*[Square().set_fill(RED, opacity=0.5) for i in range(3) for j in range(3)])
            .arrange_in_grid(3, 3, buff=0.0)
            .move_to(LEFT * d)
        )
        matrix_3x3_conv = (
            VGroup(*[Square().set_fill(DARK_BLUE, opacity=1.0) for i in range(3) for j in range(3)])
            .arrange_in_grid(3, 3, buff=0.0)
            .move_to(RIGHT * d)
        )

        for i in range(5):
            for j in range(5):
                matrix_5x5_input.add(Text(str(x[i, j])).move_to(matrix_5x5_input[i * 5 + j].get_center()))

        for i in range(3):
            for j in range(3):
                matrix_3x3_filter.add(Text(str(w[i, j])).move_to(matrix_3x3_filter[i * 3 + j].get_center()))

        self.play(*[Create(sq) for sq in matrix_3x3_conv], *[Create(sq) for sq in matrix_5x5_input])
        self.wait()

        self.play(*[Create(sq) for sq in matrix_3x3_filter])
        self.wait()

        for j in range(3):
            for i in range(3):
                self.play(matrix_3x3_filter.animate.move_to(matrix_5x5_input[(j + 1) * 5 + (i + 1)].get_center()))
                y = np.sum(x[j : j + w.shape[0], i : i + w.shape[1]] * w)
                self.add(Text(str(y)).move_to(matrix_3x3_conv[j * 3 + i].get_center()))
                self.wait()

        self.play(FadeOut(matrix_3x3_filter))
