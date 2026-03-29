# manim -pql -i rnn_forward.py

from manim import *

config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 25.0
config.frame_width = 25.0


class CustomScene(Scene):
    def construct(self):
        h_state = Circle(radius=0.4, color=BLUE, fill_opacity=0.2)
        h_state.move_to(LEFT * 8.5)
        h_label = MathTex("h", color=WHITE).move_to(h_state.get_left() - np.array([0.5, 0.0, 0.0]))
        self.add(h_state, h_label)

        v_arrow = Arrow(
            h_state.get_top(),
            h_state.get_top() + UP * 2.0,
            color=WHITE,
            max_tip_length_to_length_ratio=2.0,
        )
        v_label = MathTex("V", color=WHITE).move_to(v_arrow.get_left() - np.array([0.3, 0.1, 0.0]))
        self.add(v_arrow, v_label)

        u_arrow = Arrow(
            h_state.get_bottom() + DOWN * 2.0,
            h_state.get_bottom(),
            color=WHITE,
            max_tip_length_to_length_ratio=2.0,
        )
        u_label = MathTex("U", color=WHITE).move_to(u_arrow.get_left() - np.array([0.3, 0.1, 0.0]))
        self.add(u_arrow, u_label)

        w_loop = Arc(
            start_angle=PI + PI / 4,
            angle=PI + PI / 2,
            radius=0.7,
            arc_center=h_state.get_right() + (h_state.get_right() - h_state.get_left()) / 2,
            color=WHITE,
        )
        w_label = MathTex("W", color=WHITE).move_to(w_loop.get_center() + np.array([0.8, 0.8, 0.8]))
        self.add(w_loop, w_label)

        x_label = MathTex("x", color=WHITE).move_to(u_arrow.get_bottom() - np.array([0.0, 0.3, 0.0]))
        self.add(x_label)

        o_state = Circle(radius=0.4, color=BLUE, fill_opacity=0.2)
        o_state.move_to(v_arrow.get_end() + (v_arrow.get_end() - v_arrow.get_start()) / 2)
        o_label = MathTex("o", color=WHITE).move_to(o_state.get_left() - np.array([0.5, 0.0, 0.0]))
        self.add(o_state, o_label)

        unfold_arrow = Arrow(
            w_loop.get_center() + RIGHT * 1.0,
            w_loop.get_center() + RIGHT * 3.0,
            color=BLUE,
            stroke_width=15.0,
            max_stroke_width_to_length_ratio=10.0,
        )
        unfold_text = MathTex(r"\text{Unfold}").next_to(unfold_arrow, DOWN, buff=0.5)
        self.add(unfold_arrow, unfold_text)

        h_texts = [r"h_{t-1}", r"h_{t}", r"h_{t+1}"]
        x_texts = [r"x_{t-1}", r"x_{t}", r"x_{t+1}"]
        o_texts = [r"o_{t-1}", r"o_{t}", r"o_{t+1}"]
        for i in range(3):
            h_state = Circle(radius=0.4, color=BLUE, fill_opacity=0.2)
            h_state.move_to(LEFT * 1.0 + RIGHT * (i * 4))
            h_label = MathTex(h_texts[i], color=WHITE).move_to(h_state.get_center() + np.array([0.7, 0.7, 0.7]))
            self.add(h_state, h_label)

            v_arrow = Arrow(
                h_state.get_top(),
                h_state.get_top() + UP * 2.0,
                color=WHITE,
                max_tip_length_to_length_ratio=2.0,
            )
            v_label = MathTex("V", color=WHITE).move_to(v_arrow.get_left() - np.array([0.3, 0.1, 0.0]))
            self.add(v_arrow, v_label)

            u_arrow = Arrow(
                h_state.get_bottom() + DOWN * 2.0,
                h_state.get_bottom(),
                color=WHITE,
                max_tip_length_to_length_ratio=2.0,
            )
            u_label = MathTex("U", color=WHITE).move_to(u_arrow.get_left() - np.array([0.3, 0.1, 0.0]))
            self.add(u_arrow, u_label)

            x_label = MathTex(x_texts[i], color=WHITE).move_to(u_arrow.get_bottom() - np.array([0.0, 0.3, 0.0]))
            self.add(x_label)

            o_label = MathTex(o_texts[i], color=WHITE).move_to(v_arrow.get_top() + np.array([0.0, 0.3, 0.0]))
            self.add(o_label)

            w_arrow = Arrow(
                h_state.get_right() + LEFT * 4.0,
                h_state.get_left(),
                color=WHITE,
                max_stroke_width_to_length_ratio=2.0,
            )
            w_label = MathTex("W", color=WHITE).move_to(w_arrow.get_center() - np.array([0.0, 0.5, 0.0]))
            self.add(w_arrow, w_label)

            w_arrow = Arrow(
                h_state.get_right(),
                h_state.get_left() + RIGHT * 4.0,
                color=WHITE,
                max_stroke_width_to_length_ratio=2.0,
            )
            w_label = MathTex("W", color=WHITE).move_to(w_arrow.get_center() - np.array([0.0, 0.5, 0.0]))
            self.add(w_arrow, w_label)
