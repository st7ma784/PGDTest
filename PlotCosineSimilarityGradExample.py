import numpy as np
import torch
import matplotlib.pyplot as plt

# Generate 64 random points

random_points = np.random.rand(64, 2)

# Define target point and prediction point
target_point = np.array([0.5, 0.5])
prediction_point = np.array([0.7, 0.7])

# Generate a ring of points around the prediction point
theta = np.linspace(0, 2 * np.pi, 100)
radius = 0.1
ring_points = np.array([prediction_point + radius * np.array([np.cos(t), np.sin(t)]) for t in theta])

# Calculate cross entropy loss for each point in the ring

normed_ring_points = ring_points / torch.norm(torch.tensor(ring_points), dim=-1, keepdim=True)
alltargets= torch.tensor(np.cat([target_point, random_points]))
normed_alltargets = alltargets / torch.norm(torch.tensor(alltargets), dim=-1, keepdim=True)
cosine_similarity = normed_ring_points @ normed_alltargets.T
losses = torch.nn.CrossEntropyLoss()(torch.tensor(ring_points), torch.zeros(65))

# Plot the random points
plt.scatter(random_points[:, 0], random_points[:, 1], label='Random Points')

# Plot the target point
plt.scatter(target_point[0], target_point[1], color='red', label='Target Point')

# Plot the prediction point
plt.scatter(prediction_point[0], prediction_point[1], color='blue', label='Prediction Point')

# Plot the ring points with color based on cross entropy loss
norm = plt.Normalize(losses.min(), losses.max())
colors = plt.cm.viridis(norm(losses))
plt.scatter(ring_points[:, 0], ring_points[:, 1], c=colors, label='Ring Points')

# Add color bar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Cross Entropy Loss')

# Add legend and show plot
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Cosine Similarity and Cross Entropy Loss Visualization')
plt.show()