import unittest
import numpy as np
import edunn as nn
from edunn.model import class_counter, model_name_registry

class TestTrainers(unittest.TestCase):
    def test_supervised_trainer(self):
        # Linear regression task
        x = np.random.randn(100, 2)
        true_w = np.array([[1.5], [-2.0]])
        y = x @ true_w + 0.5
        
        model = nn.Linear(2, 1)
        # Initialize bias to 0 for simplicity
        params = model.get_parameters()
        for k in params:
            if 'b' in k:
                params[k][:] = 0
        
        optimizer = nn.GradientDescent(lr=0.01)
        error = nn.MeanError(nn.SquaredError())
        
        trainer = nn.SupervisedTrainer(model, optimizer, error, epochs=50, batch_size=10)
        history = trainer.train(x, y, verbose=False)
        
        self.assertEqual(len(history), 50)
        self.assertLess(history[-1], history[0])

    def test_sequence_trainer_accumulation(self):
        # Gradient accumulation should work
        x = np.random.randn(100, 2)
        y = np.random.randn(100, 1)
        
        model = nn.Linear(2, 1)
        optimizer = nn.GradientDescent(lr=0.01)
        error = nn.MeanError(nn.SquaredError())
        
        # Accumulate over 5 steps
        trainer = nn.SequenceTrainer(model, optimizer, error, epochs=2, batch_size=10, accumulation_steps=5)
        
        history = trainer.train(x, y, verbose=False)
        self.assertEqual(len(history), 2)

    def test_accumulation_equivalence(self):
        # Accumulation over N steps with batch_size B 
        # should be roughly equivalent to batch_size N*B (if no shuffling)
        
        x = np.random.randn(100, 2)
        y = np.random.randn(100, 1)
        
        def train_config(bs, acc):
            # Reset registry to have predictable names
            class_counter.clear()
            model_name_registry.clear()
            
            model = nn.Linear(2, 1)
            params = model.get_parameters()
            # Set fixed initial values
            for k in list(params.keys()):
                if 'w' in k:
                    params[k][:] = 0.1
                if 'b' in k:
                    params[k][:] = 0.0
            
            opt = nn.GradientDescent(lr=0.01)
            err = nn.MeanError(nn.SquaredError())
            trainer = nn.SequenceTrainer(model, opt, err, epochs=1, batch_size=bs, shuffle=False, accumulation_steps=acc)
            trainer.train(x, y, verbose=False)
            
            # Find the weight matrix again
            for k in params:
                if 'w' in k:
                    return np.array(params[k]).copy()

        # BS=20, ACC=1
        w1 = train_config(20, 1)
        w2 = train_config(10, 2)
        
        self.assertIsNotNone(w1, "Weight matrix w1 not found")
        self.assertIsNotNone(w2, "Weight matrix w2 not found")
        
        # They should be very close
        np.testing.assert_allclose(w1, w2, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
