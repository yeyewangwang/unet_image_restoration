import torch
import argparse


def get_model():
    # Define and return your neural network model here
    # Example: return MyNeuralNetworkModel()
    pass


def get_loss_function():
    # Define and return your loss function here
    # Example: return torch.nn.CrossEntropyLoss()
    pass


def get_input_data():
    # Define and return your input data here
    # Example: return torch.randn(10, 3, 32, 32)  # Dummy input data
    pass


def get_target_data():
    # Define and return your target data here
    # Example: return torch.randint(0, 10, (10,))  # Dummy target data
    pass


def compute_numerical_gradient(model, loss_function, input,
                               target, epsilon=1e-5):
    numerical_gradients = []

    for param in model.parameters():
        param_data = param.data
        param_grad = torch.zeros_like(param_data)

        for i in range(param_data.numel()):
            original_value = param_data.view(-1)[i].item()

            # Compute J(theta + epsilon)
            param_data.view(-1)[
                i] = original_value + epsilon
            output = model(input)
            loss1 = loss_function(output, target)

            # Compute J(theta - epsilon)
            param_data.view(-1)[
                i] = original_value - epsilon
            output = model(input)
            loss2 = loss_function(output, target)

            # Reset the parameter value
            param_data.view(-1)[i] = original_value

            # Compute the numerical gradient
            grad = (loss1 - loss2) / (2 * epsilon)
            param_grad.view(-1)[i] = grad

        numerical_gradients.append(param_grad)

    return numerical_gradients


def main(epsilon):
    model = get_model()
    loss_function = get_loss_function()
    input = get_input_data()
    target = get_target_data()

    # Perform forward pass and compute analytical gradients
    model.zero_grad()
    output = model(input)
    loss = loss_function(output, target)
    loss.backward()

    # Compute numerical gradients
    numerical_gradients = compute_numerical_gradient(model,
                                                     loss_function,
                                                     input,
                                                     target,
                                                     epsilon)

    # Compare the analytical and numerical gradients
    for i, param in enumerate(model.parameters()):
        analytical_grad = param.grad
        numerical_grad = numerical_gradients[i]

        # Check if gradients are close
        assert torch.allclose(analytical_grad,
                              numerical_grad,
                              atol=1e-4), f"Gradients do not match for parameter {i}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute numerical gradients and compare with analytical gradients.')
    parser.add_argument('--epsilon', type=float,
                        default=1e-5,
                        help='Epsilon value for numerical gradient computation')
    args = parser.parse_args()

    main(args.epsilon)
