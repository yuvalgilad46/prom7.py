import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = """
0 , 0 , 1
1 , 0 , 2
2 , 0 , 3
3 , 0 , 2
4 , 0 , 4
5 , 0 , 4
6 , 0 , 6
7 , 0 , 6
8 , 0 , 7
9 , 0 , 10
10 , 0 , 7
11 , 0 , 10
12 , 0 , 7
13 , 0 , 6
14 , 0 , 6
15 , 0 , 4
16 , 0 , 4
17 , 0 , 2
18 , 0 , 3
19 , 0 , 2

"""

# Split into lines and build the array
array = []
for line in data.strip().splitlines():
    parts = [int(x.strip()) for x in line.split(',')]
    array.append(parts)
print(array)
print(array[2][2])


data = array

# Separate into x, y, z lists
x = [point[0] for point in data]
y = [point[1] for point in data]
z = [point[2] for point in data]

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, c=z, cmap='inferno', s=50)

# Connect the points in order
# ax.plot(x, y, z, color='black', linewidth=1)

# Axis labels and title
ax.set_xlabel('position change')
ax.set_ylabel('velocity change')
ax.set_zlabel('number of pins fallen')
ax.set_title('pins vs. position and velocity change')

plt.show()
