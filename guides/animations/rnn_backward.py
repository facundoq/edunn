# manim -pql -i rnn_backward.py

from manim import *

config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 25.0
config.frame_width = 25.0


class CustomScene(Scene):
    def construct(self):
        n_states = 5

        # states
        positions = [LEFT * n_states + RIGHT * i * 2.5 for i in range(n_states)]
        states = []
        for i, pos in enumerate(positions):
            node = Circle(radius=0.4, color=BLUE, fill_opacity=0.2).move_to(pos)
            state_label = MathTex(f"s_{i}", color=WHITE).move_to(pos)
            x_label = MathTex(f"x_{i}", color=WHITE).next_to(node, DOWN * 4, buff=0.5)
            E_label = MathTex(f"E_{i}", color=WHITE).next_to(node, UP * 4, buff=0.5)
            self.add(node, state_label, x_label, E_label)
            states.append({"node": node, "state": state_label, "x": x_label, "E": E_label})

        states.append(states[0])  # dummy

        # arrows (forward)
        for i in range(n_states):
            # horizontal s_i -> s_i+1
            arrow = Arrow(
                states[i]["node"].get_right() + DOWN * 0.2,
                states[i + 1]["node"].get_left() + DOWN * 0.2,
                buff=0.1,
                color=WHITE,
                max_tip_length_to_length_ratio=0.15,
            )
            self.add(arrow) if i != (n_states - 1) else None
            # vertical s_i -> E_i
            arrow = Arrow(
                states[i]["node"].get_top() + LEFT * 0.2,
                states[i]["E"].get_bottom() + LEFT * 0.2,
                color=WHITE,
                max_tip_length_to_length_ratio=0.15,
            )
            self.add(arrow)
            # vertical x_i -> s_i
            arrow = Arrow(
                states[i]["x"].get_top() + LEFT * 0.2,
                states[i]["node"].get_bottom() + LEFT * 0.2,
                color=WHITE,
                max_tip_length_to_length_ratio=0.15,
            )
            self.add(arrow)

        # arrows (backward)
        for i in range(n_states):
            # horizontal s_i <- s_i+1
            grad_arrow = Arrow(
                states[i]["node"].get_left() + UP * 0.2,
                states[i - 1]["node"].get_right() + UP * 0.2,
                buff=0.1,
                color=RED,
                max_tip_length_to_length_ratio=0.15,
            )
            grad_label = (
                MathTex(f"\\frac{{\\partial s_{i}}}{{\\partial s_{i - 1}}}", color=RED)
                .scale(0.6)
                .next_to(grad_arrow, UP, buff=0.1)
            )
            self.add(grad_arrow, grad_label) if i != 0 else None
            # vertical s_i <- E_i
            grad_arrow = Arrow(
                states[i]["E"].get_bottom() + RIGHT * 0.2,
                states[i]["node"].get_top() + RIGHT * 0.2,
                color=RED,
                max_tip_length_to_length_ratio=0.15,
            )
            grad_label = (
                MathTex(f"\\frac{{\\partial E_{i}}}{{\\partial s_{i}}}", color=RED)
                .scale(0.6)
                .next_to(grad_arrow, RIGHT, buff=0.1)
            )
            self.add(grad_arrow, grad_label)
            # vertical x_i <- s_i
            grad_arrow = Arrow(
                states[i]["node"].get_bottom() + RIGHT * 0.2,
                states[i]["x"].get_top() + RIGHT * 0.2,
                color=RED,
                max_tip_length_to_length_ratio=0.15,
            )
            grad_label = (
                MathTex(f"\\frac{{\\partial E_{i}}}{{\\partial x_{i}}}", color=RED)
                .scale(0.6)
                .next_to(grad_arrow, RIGHT, buff=0.1)
            )
            self.add(grad_arrow, grad_label)
