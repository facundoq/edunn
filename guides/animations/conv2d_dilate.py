# manim -pql -i conv2d_dilate.py

from manim import *

config.pixel_height = 900
config.pixel_width = 1900
config.frame_height = 25.0
config.frame_width = 25.0


class CustomScene(Scene):
    def construct(self):
        matrix_2x2_input = VGroup(
            *[Square().set_fill(DARK_BLUE, opacity=1) for i in range(2) for j in range(2)]
        ).arrange_in_grid(2, 2, buff=0.0)
        matrix_3x3_dilate = VGroup(
            *[Square().set_fill(GRAY, opacity=1) for i in range(3) for j in range(3)]
        ).arrange_in_grid(3, 3, buff=0.0)

        dEdy = [
            [r"\frac{\partial E}{\partial " + "y_{}".format("{" + f"{i + 1}{j + 1}" + "}") + "}" for i in range(2)]
            for j in range(2)
        ]
        dEdy = list(map(list, zip(*dEdy)))  # https://stackoverflow.com/a/6473724/11975664
        text_dilate = [["0" for _ in range(3)] for _ in range(3)]

        self.play(Create(matrix_2x2_input))
        self.wait()

        for i in range(2):
            for j in range(2):
                eq = MathTex(str(dEdy[i][j])).scale(1).set_color_by_tex("x", BLACK)
                matrix_2x2_input[2 * i + j].add(eq.move_to(np.array([-1, 1, 0]) + np.array([j * 2, -i * 2, 0])))
        self.play(matrix_2x2_input.animate.arrange_in_grid(2, 2, buff=2))

        for i in range(3):
            for j in range(3):
                eq = MathTex(str(text_dilate[i][j])).scale(1).set_color_by_tex("x", BLACK)
                matrix_3x3_dilate[3 * i + j].add(eq.move_to(np.array([-2, 2, 0]) + np.array([j * 2, -i * 2, 0])))
        matrix_3x3_dilate.set_z_index(matrix_2x2_input.z_index - 1)  # https://stackoverflow.com/a/69668382/189270
        self.play(Create(matrix_3x3_dilate))

        dots = VGroup(*[Dot(np.array([-1, 4, 0])), Dot(np.array([1, 4, 0]))])
        line = Line(dots[0], dots[1])
        text = Tex(r"dilation=S-1", color=WHITE, font_size=48)

        self.add(dots)
        self.play(Create(line))
        self.add(text.move_to(np.array([0, 5, 0])))
        self.wait()
