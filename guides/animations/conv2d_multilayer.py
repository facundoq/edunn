# manim -pql -i conv2d_multilayer.py

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
                Convolutional2DLayer(1, 7, 3, filter_spacing=0.32),
                Convolutional2DLayer(3, 5, 3, filter_spacing=0.32),
                Convolutional2DLayer(5, 3, 3, filter_spacing=0.18),
            ],
            layer_spacing=0.25,
        )
        nn.move_to(ORIGIN)
        self.add(nn)
        forward_pass = nn.make_forward_pass_animation()
        self.play(forward_pass, run_time=10)
