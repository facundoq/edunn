# manim -pql -i conv2d_padding.py

from manim import *
from manim_ml.neural_network import NeuralNetwork, Convolutional2DLayer

config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0


class CustomScene(ThreeDScene):
    def construct(self):
        nn = NeuralNetwork(
            [
                Convolutional2DLayer(
                    num_feature_maps=1,
                    feature_map_size=5,
                    padding=1,
                    padding_dashed=True,
                    filter_spacing=0.32,
                    show_grid_lines=True,
                ),
                Convolutional2DLayer(
                    num_feature_maps=1,
                    feature_map_size=5,
                    filter_size=3,
                    padding=0,
                    padding_dashed=False,
                    filter_spacing=0.32,
                    show_grid_lines=True,
                ),
            ],
            layer_spacing=1.25,
        )
        nn.move_to(ORIGIN)
        self.add(nn)
        forward_pass = nn.make_forward_pass_animation()
        self.play(forward_pass, run_time=5)
