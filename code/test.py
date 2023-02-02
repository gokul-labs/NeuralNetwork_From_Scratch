from neural_network import MLP

nn = MLP(3, [4, 4, 1])

X = [[2.0, 4.0, -8.0],
     [-6.0, 3.0, 5.0],
     [7.0, 2.0, 7.0],
     [8.0, 9.0, -4.0]]
Y = [1.0, -1.0, -1.0, 1.0]
Y_pred = [nn(x)[0] for x in X]
print(Y_pred)

LEARNING_RATE = 0.01
EPOCHS = 25

for epoch in range(EPOCHS):

    # Forward pass
    Y_pred = [nn(x)[0] for x in X]
    loss = sum([(Y_pred[i] - Y[i]) ** 2 for i in range(len(Y))])

    # Zero-grad
    for p in nn.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Update
    for p in nn.parameters():
        p.val += -LEARNING_RATE * p.grad

    print(epoch, loss)

print(Y_pred)
