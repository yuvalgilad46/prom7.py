# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import traceback
from matplotlib.widgets import Button

# Removed scipy.differentiate import as it wasn't used

# --- Quaternion Class (Mostly Unchanged) ---
class Quaternion:
    """Represents a quaternion for 3D rotations."""
    def __init__(self, w, x, y, z):
        # Ensure components are floats for consistency
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        # Corrected f-string formatting and typo
        return f"Q(w={self.w:.3f}, x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

    def __mul__(self, other):
        """Quaternion multiplication or scalar multiplication."""
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float, np.number)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            # Allow multiplication by numpy arrays component-wise (useful for derivatives)
            if isinstance(other, (np.ndarray, list)) and len(other) == 4:
                 return Quaternion(self.w * other[0], self.x * other[1], self.y * other[2], self.z * other[3])
            raise TypeError(f"Multiplication only supports Quaternion, scalar, or 4-element array/list, not {type(other)}")

    def __add__(self, other):
        """Quaternion addition."""
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        # Allow addition with 4-element lists/arrays (useful for RK4 steps)
        elif isinstance(other, (np.ndarray, list)) and len(other) == 4:
             return Quaternion(self.w + other[0], self.x + other[1], self.y + other[2], self.z + other[3])
        else:
            raise TypeError(f"Addition only supports two Quaternions or Quaternion and 4-element array/list, not {type(other)}")

    def conjugate(self):
        """Returns the conjugate of the quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_sq(self):
        """Calculates the squared norm (magnitude squared)."""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def norm(self):
        """Calculates the norm (magnitude) of the quaternion."""
        return np.sqrt(self.norm_sq())

    def normalize(self):
        """Normalizes the quaternion to unit length."""
        n = self.norm()
        if n < 1e-9:
            # print("Warning: Normalizing near-zero quaternion. Returning identity.")
            return Quaternion.identity() # Use static method
        inv_n = 1.0 / n
        self.w *= inv_n
        self.x *= inv_n
        self.y *= inv_n
        self.z *= inv_n
        return self # Return self for chaining if needed, but modifies in place

    def to_rotation_matrix(self):
        """Converts the unit quaternion to a 3x3 rotation matrix."""
        # It's assumed the quaternion is already normalized for physics updates
        w, x, y, z = self.w, self.x, self.y, self.z
        x2, y2, z2 = x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        R = np.array([
            [1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * xz + 2 * wy],
            [2 * xy + 2 * wz, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx],
            [2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x2 - 2 * y2]
        ], dtype=float) # Ensure float type
        return R

    def rotate_vector(self, v):
        """Rotates a 3D vector using the quaternion (q * v * q_conj)."""
        v = np.asarray(v, dtype=float)
        if v.shape != (3,):
            raise ValueError("Input vector must be 3D")
        v_quat = Quaternion(0.0, v[0], v[1], v[2])
        # Assume normalized for performance, or uncomment below
        # q_norm = self.normalize() # Creates copy if normalize returns new Q
        rotated_v_quat = self * v_quat * self.conjugate()
        return np.array([rotated_v_quat.x, rotated_v_quat.y, rotated_v_quat.z])

    def to_list(self):
         """Return quaternion components as a list."""
         return [self.w, self.x, self.y, self.z]

    @staticmethod
    def identity():
        """Returns the identity quaternion."""
        return Quaternion(1.0, 0.0, 0.0, 0.0)

# --- Base Rigid Body Class ---
class RigidBody:
    def __init__(self, obj_id, obj_type, pos, vel, quat, ang_vel, mass):
        self.id = obj_id # Unique identifier
        self.obj_type = obj_type # 'sphere', 'cylinder', etc.
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.quat = quat if isinstance(quat, Quaternion) else Quaternion(*quat)
        self.ang_vel = np.asarray(ang_vel, dtype=float) # World frame
        self.mass = float(mass)
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0

        # Inertia properties to be set by subclasses
        self.inertia_local = np.zeros((3, 3))
        self.inv_inertia_local = np.zeros((3, 3))
        self.inv_inertia_world = np.zeros((3, 3))
        self.update_inv_inertia_world() # Initial calculation

    def update_inv_inertia_world(self):
        """Calculates the inverse inertia tensor in world coordinates"""
        R = self.quat.to_rotation_matrix()
        self.inv_inertia_world = R @ self.inv_inertia_local @ R.T

    def get_state_vector(self):
        """Return current state as a numpy vector (pos, vel, quat_list, ang_vel)."""
        return np.concatenate([
            self.pos,
            self.vel,
            self.quat.to_list(),
            self.ang_vel
        ])

    def set_state_from_vector(self, vec):
        """Update state from a numpy vector."""
        self.pos = vec[0:3]
        self.vel = vec[3:6]
        self.quat = Quaternion(*vec[6:10])
        self.ang_vel = vec[10:13]
        # Ensure quaternion is normalized after update
        self.quat.normalize()
        # Update world inertia tensor as orientation changed
        self.update_inv_inertia_world()

    def __repr__(self):
        # Base representation
         return f"{self.obj_type.capitalize()}(id={self.id}, pos={self.pos}, vel={self.vel}, quat={self.quat}, ang_vel={self.ang_vel})"

# --- Ball Class ---
class Ball(RigidBody):
    def __init__(self, obj_id, pos, vel, quat, ang_vel, radius, mass):
        super().__init__(obj_id, 'sphere', pos, vel, quat, ang_vel, mass)
        self.radius = float(radius)
        self.height = float(radius)
        self._calculate_inertia()
        self.update_inv_inertia_world() # Recalculate with correct local inertia

    def _calculate_inertia(self):
        if self.mass > 0 and self.radius > 0:
            I_scalar = (2.0 / 5.0) * self.mass * self.radius**2
            self.inertia_local = np.diag([I_scalar] * 3)
            self.inv_inertia_local = np.diag([1.0 / I_scalar] * 3)
        else:
            self.inertia_local = np.eye(3) # Prevent division by zero
            self.inv_inertia_local = np.eye(3)

    # Override repr for specific details if needed
    # def __repr__(self):
    #     base_repr = super().__repr__()
    #     return f"{base_repr[:-1]}, radius={self.radius})"


# --- Pin Class ---
class Pin(RigidBody):
    def __init__(self, obj_id, pos, vel, quat, ang_vel, radius, height, mass):
        super().__init__(obj_id, 'cylinder', pos, vel, quat, ang_vel, mass)
        self.radius = float(radius)
        self.height = float(height)
        self._calculate_inertia()
        self.update_inv_inertia_world() # Recalculate with correct local inertia

    def _calculate_inertia(self):
        if self.mass > 0 and self.radius > 0 and self.height > 0:
            Ixy = (1.0 / 12.0) * self.mass * (3 * self.radius**2 + self.height**2)
            Iz = (1.0 / 2.0) * self.mass * self.radius**2
            self.inertia_local = np.diag([Ixy, Ixy, Iz])
            try:
                self.inv_inertia_local = np.linalg.inv(self.inertia_local)
            except np.linalg.LinAlgError:
                print(f"Warning: Pin {self.id} inertia tensor is singular. Using pseudo-inverse.")
                self.inv_inertia_local = np.linalg.pinv(self.inertia_local)
        else:
             self.inertia_local = np.eye(3)
             self.inv_inertia_local = np.eye(3) # Prevent issues

    def get_vertices(self, n_sides=12):
        """ Calculates world coordinates of top and bottom cap vertices. """
        R = self.quat.to_rotation_matrix()
        vertices = {'top': [], 'bottom': []}
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides
            # Local coordinates (relative to center of mass)
            local_top = np.array([self.radius * np.cos(angle), self.radius * np.sin(angle), self.height / 2.0])
            local_bottom = np.array([self.radius * np.cos(angle), self.radius * np.sin(angle), -self.height / 2.0])
            # Convert to world coordinates
            vertices['top'].append(self.pos + R @ local_top)
            vertices['bottom'].append(self.pos + R @ local_bottom)
        return vertices


# --- Simulation Parameters ---
DT = 0.01
NUM_FRAMES = 500 # Increased frames
COLLISION_EPSILON = 0.8 # Slightly less bouncy
GRAVITY = np.array([0.0, 0.0, -9.81])
FLOOR_RESTITUTION = 0.3 # How much bounce off the floor
FLOOR_FRICTION_COEFF = 0.5 # Coefficient of kinetic friction

is_paused = False

# --- Initial States & Objects ---
ball_1 = Ball(obj_id='ball_1',
              pos=np.array([-1.5, 0.0, 0.109]), # Start touching floor
              vel=np.array([10.0, 0.1, 0.0]),    # Faster initial speed
              quat=Quaternion.identity(),
              ang_vel=np.array([0.0, 0.0, 0.0]), # Maybe add initial spin later
              radius=0.109,
              mass= 5.0)

pins = []
# Standard 10-pin setup (approximate positions relative to origin (0,0))
# Distances are roughly: Row separation ~ sqrt(3)*PinDiam, Pin separation in row ~ 1.0*PinDiam
pin_diam = 2 * 0.06 # Pin diameter
row_sep = np.sqrt(3) * 0.5 * (pin_diam * 2.2) # sqrt(3)/2 * spacing
pin_spacing = pin_diam * 1.1 # Spacing within a row

pin_positions = [
    # Row 1
    np.array([0.0, 0.0, 0.0]), # Pin 1 (at origin for simplicity)
    # # Row 2
    np.array([row_sep, pin_spacing / 2.0, 0.0]), # Pin 2
    np.array([row_sep, -pin_spacing / 2.0, 0.0]), # Pin 3
    # Row 3
    np.array([2 * row_sep, pin_spacing, 0.0]), # Pin 4
    np.array([2 * row_sep, 0.0, 0.0]), # Pin 5
    np.array([2 * row_sep, -pin_spacing, 0.0]), # Pin 6
    # Row 4
    np.array([3 * row_sep, pin_spacing * 1.5, 0.0]), # Pin 7
    np.array([3 * row_sep, pin_spacing * 0.5, 0.0]), # Pin 8
    np.array([3 * row_sep, -pin_spacing * 0.5, 0.0]), # Pin 9
    np.array([3 * row_sep, -pin_spacing * 1.5, 0.0]), # Pin 10
]

for i, pos in enumerate(pin_positions):
     # Adjust Z so bottom is at 0
    initial_pin_pos = pos + np.array([0.0, 0.0, 0.38 / 2.0])
    pins.append(Pin(obj_id=f'pin_{i+1}',
                   pos=initial_pin_pos,
                   vel=np.array([0.0, 0.0, 0.0]),
                   quat=Quaternion.identity(),
                   ang_vel=np.array([0.0, 0.0, 0.0]),
                   radius=0.06,
                   height=0.38,
                   mass=1.5))

# Combine all simulation objects
sim_objects = [ball_1] + pins

# --- Physics Engine ---

# Structure to hold derivatives for one object
class Derivatives:
    def __init__(self):
        self.d_pos = np.zeros(3)
        self.d_vel = np.zeros(3)
        self.d_quat = np.zeros(4) # Store as list/array for calculations
        self.d_ang_vel = np.zeros(3)

def calculate_derivatives(obj):
    """ Calculates time derivatives for a single RigidBody object. """
    derivs = Derivatives()
    derivs.d_pos = obj.vel

    # --- Forces ---
    # Gravity
    force = GRAVITY * obj.mass
    # Could add other forces here (drag, etc.)

    derivs.d_vel = force * obj.inv_mass

    # --- Torques ---
    # Currently no external torques applied
    torque = np.zeros(3)

    # Angular acceleration = I_world^-1 * (torque - omega x (I_world * omega))
    # The omega x (I * omega) term is important for non-spherical objects but often omitted in simpler engines.
    # Let's omit it for now for simplicity, similar to the original code.
    # Re-calculate world inertia just in case? Should be updated after quat change.
    # obj.update_inv_inertia_world() # Might be redundant if RK4 updates it
    derivs.d_ang_vel = obj.inv_inertia_world @ torque

    # Quaternion derivative: dQ/dt = 0.5 * Quaternion(0, omega) * Q
    omega_q = Quaternion(0.0, *obj.ang_vel)
    q_dot = omega_q * obj.quat * 0.5
    derivs.d_quat = q_dot.to_list()

    return derivs


def rk4_step(obj, dt):
    """ Performs a single RK4 step for one RigidBody object. """
    # Store initial state
    y0_pos = obj.pos.copy()
    y0_vel = obj.vel.copy()
    # Make sure y0_quat is a distinct Quaternion object for the start of the step
    # If obj.quat is modified in place by normalize, this reference is fine.
    y0_quat = obj.quat
    y0_ang_vel = obj.ang_vel.copy()

    # --- Calculate k1 ---
    k1 = calculate_derivatives(obj)
    # Convert d_quat list to NumPy array for element-wise operations
    k1_d_quat_np = np.array(k1.d_quat)

    # --- Calculate k2 state ---
    # Create a temporary quaternion for intermediate state calculation
    temp_quat_k2 = y0_quat + k1_d_quat_np * (dt / 2.0) # Uses Quaternion.__add__(np.array) overload
    temp_quat_k2.normalize()
    # Create a temporary object state for derivative calculation (or update obj temporarily)
    obj_temp_k2 = RigidBody(obj.id, obj.obj_type, # Create a shallow copy or update temporarily
                           y0_pos + k1.d_pos * (dt / 2.0),
                           y0_vel + k1.d_vel * (dt / 2.0),
                           temp_quat_k2, # Use normalized intermediate quat
                           y0_ang_vel + k1.d_ang_vel * (dt / 2.0),
                           obj.mass)
    # Copy necessary attributes if needed (like inertia, radius etc.)
    obj_temp_k2.inv_inertia_local = obj.inv_inertia_local # Need local inertia
    obj_temp_k2.update_inv_inertia_world() # Update world based on temp orientation
    k2 = calculate_derivatives(obj_temp_k2)
    k2_d_quat_np = np.array(k2.d_quat)

    # --- Calculate k3 state ---
    temp_quat_k3 = y0_quat + k2_d_quat_np * (dt / 2.0)
    temp_quat_k3.normalize()
    obj_temp_k3 = RigidBody(obj.id, obj.obj_type,
                           y0_pos + k2.d_pos * (dt / 2.0),
                           y0_vel + k2.d_vel * (dt / 2.0),
                           temp_quat_k3,
                           y0_ang_vel + k2.d_ang_vel * (dt / 2.0),
                           obj.mass)
    obj_temp_k3.inv_inertia_local = obj.inv_inertia_local
    obj_temp_k3.update_inv_inertia_world()
    k3 = calculate_derivatives(obj_temp_k3)
    k3_d_quat_np = np.array(k3.d_quat)

    # --- Calculate k4 state ---
    temp_quat_k4 = y0_quat + k3_d_quat_np * dt
    temp_quat_k4.normalize()
    obj_temp_k4 = RigidBody(obj.id, obj.obj_type,
                           y0_pos + k3.d_pos * dt,
                           y0_vel + k3.d_vel * dt,
                           temp_quat_k4,
                           y0_ang_vel + k3.d_ang_vel * dt,
                           obj.mass)
    obj_temp_k4.inv_inertia_local = obj.inv_inertia_local
    obj_temp_k4.update_inv_inertia_world()
    k4 = calculate_derivatives(obj_temp_k4)
    k4_d_quat_np = np.array(k4.d_quat)

    # --- Combine and Update Object's Actual State ---
    obj.pos = y0_pos + (dt / 6.0) * (k1.d_pos + 2*k2.d_pos + 2*k3.d_pos + k4.d_pos)
    obj.vel = y0_vel + (dt / 6.0) * (k1.d_vel + 2*k2.d_vel + 2*k3.d_vel + k4.d_vel)

    # Quaternion update: Use weighted average of derivative numpy arrays
    final_d_quat_np = (k1_d_quat_np + 2*k2_d_quat_np + 2*k3_d_quat_np + k4_d_quat_np) / 6.0
    # Add the final scaled delta to the original quaternion
    obj.quat = y0_quat + final_d_quat_np * dt
    obj.quat.normalize() # Final normalization of the object's quaternion

    obj.ang_vel = y0_ang_vel + (dt / 6.0) * (k1.d_ang_vel + 2*k2.d_ang_vel + 2*k3.d_ang_vel + k4.d_ang_vel)

    # Final update of the object's world inertia tensor based on its new orientation
    obj.update_inv_inertia_world()


def check_and_resolve_collision(obj1, obj2):
    """ Checks and resolves collisions between two RigidBody objects. """
    # Currently only implements Ball-Pin collision
    ball = None
    pin = None
    if isinstance(obj1, Pin) and isinstance(obj2, Pin):
        pin1 = obj1
        pin2 = obj2
        pin2_pos, pin2_vel, pin2_q, pin2_ang_vel = pin2.pos, pin2.vel, pin2.quat, pin2.ang_vel
        pin1_pos, pin1_vel, pin1_q, pin1_ang_vel = pin1.pos, pin1.vel, pin1.quat, pin1.ang_vel

        vec_pin1_to_pin2 = pin2_pos - pin1_pos
        dist_centers_sq = np.dot(vec_pin1_to_pin2, vec_pin1_to_pin2)

        # Broad Phase
        bounding_radius_pin1 = np.sqrt(pin1.radius ** 2 + (pin1.height / 2) ** 2)
        min_dist_centers = pin2.radius + bounding_radius_pin1
        if dist_centers_sq > min_dist_centers ** 2:
            return  # Too far apart

        # Detailed Check
        pin1_z_axis_world = pin1_q.rotate_vector([0, 0, 1])
        pin1_y_axis_world = pin1_q.rotate_vector([0, 1, 0])
        pin1_x_axis_world = pin1_q.rotate_vector([1, 0, 0])

        proj_len = np.dot(vec_pin1_to_pin2, pin1_z_axis_world)
        closest_pt_on_axis = pin1_pos + proj_len * pin1_z_axis_world
        vec_axis_to_pin2 = pin2_pos - closest_pt_on_axis
        dist_axis_to_pin2_sq = np.dot(vec_axis_to_pin2, vec_axis_to_pin2)
        dist_axis_to_pin2 = np.sqrt(dist_axis_to_pin2_sq) if dist_axis_to_pin2_sq > 0 else 0
        is_within_height_range = abs(proj_len) <= pin1.height / 2.0
        # Use slightly tolerance for range check robustness
        radial_thresh = pin1.radius + pin2.radius
        is_within_radial_range = dist_axis_to_pin2 < radial_thresh + 1e-5

        is_within_cap_height_range = pin1.height / 2.0 < abs(proj_len) < pin1.height / 2.0 + pin2.radius + 1e-5
        is_radially_inside_cap_proj = dist_axis_to_pin2 < pin1.radius + 1e-5  # pin2 center needs project inside pin1 radius

        collided = False
        penetration_depth = 0.0
        contact_normal = np.zeros(3)
        contact_point_pin2_local = np.zeros(3)
        contact_point_pin1_local = np.zeros(3)
        contact_point_pin1_world = np.zeros(3)  # Store world point for calc

        # Check Wall Collision
        if is_within_height_range and is_within_radial_range:
            penetration_candidate = radial_thresh - dist_axis_to_pin2
            if penetration_candidate > -1e-5:  # Allow very slight overlap before trigger
                if dist_axis_to_pin2 > 1e-6:
                    contact_normal = vec_axis_to_pin2 / dist_axis_to_pin2
                else:  # pin2 center is on the pin1 axis - push radially
                    contact_normal = pin1_x_axis_world if abs(np.dot(vec_pin1_to_pin2, pin1_x_axis_world)) > abs(
                        np.dot(vec_pin1_to_pin2, pin1_y_axis_world)) else pin1_y_axis_world

                penetration_depth = max(0, penetration_candidate)  # Ensure non-negative
                collided = True
                contact_point_pin2_local = -contact_normal * pin2.radius  # Relative to pin2 center
                contact_point_pin1_world = closest_pt_on_axis + contact_normal * pin1.radius  # On pin1 surface
                contact_point_pin1_local = pin1_q.conjugate().rotate_vector(contact_point_pin1_world - pin1_pos)
                # print(f"Wall Hit: depth={penetration_depth:.4f}")

        # Check Cap Collision (only if not already hit wall)
        # Use fixed cap contact point logic!
        if not collided and is_within_cap_height_range and is_radially_inside_cap_proj:
            penetration_candidate = pin2.radius - (abs(proj_len) - pin1.height / 2.0)
            if penetration_candidate > -1e-5:
                is_top_cap = proj_len > 0
                contact_normal = pin1_z_axis_world if is_top_cap else -pin1_z_axis_world
                penetration_depth = max(0, penetration_candidate)
                collided = True

                # --- Corrected Cap Contact Point ---
                cap_center_world = pin1_pos + pin1_z_axis_world * (pin1.height / 2.0 * np.sign(proj_len))
                # vec_axis_to_pin2 is the vector from axis to pin2 center (in cap plane)
                contact_point_pin1_world = cap_center_world + vec_axis_to_pin2
                # --- End Correction ---

                contact_point_pin2_local = -contact_normal * pin2.radius
                contact_point_pin1_local = pin1_q.conjugate().rotate_vector(contact_point_pin1_world - pin1_pos)
                # print(f"Cap Hit: depth={penetration_depth:.4f}")

        if not collided or penetration_depth <= 0:
            return  # No actual collision/penetration

        # --- Resolve Penetration (move objects apart based on mass) ---
        total_mass = pin2.mass + pin1.mass
        inv_total_mass = 1.0 / total_mass if total_mass > 1e-9 else 0
        move_fraction_pin2 = pin1.mass * inv_total_mass
        move_fraction_pin1 = pin2.mass * inv_total_mass

        # Correction factor slightly > 1 to ensure separation
        correction_factor = 1.01
        correction = contact_normal * penetration_depth * correction_factor
        pin2.pos += correction * move_fraction_pin2
        pin1.pos -= correction * move_fraction_pin1

        # --- Calculate Impulse (velocity change) ---
        r_pin2_world = pin2_q.rotate_vector(contact_point_pin2_local)  # Vector from pin2 CM to contact
        r_pin1_world = pin1_q.rotate_vector(contact_point_pin1_local)  # Vector from pin1 CM to contact

        v_contact_pin2 = pin2.vel + np.cross(pin2.ang_vel, r_pin2_world)
        v_contact_pin1 = pin1.vel + np.cross(pin1.ang_vel, r_pin1_world)
        v_relative = v_contact_pin2 - v_contact_pin1
        v_rel_normal = np.dot(v_relative, contact_normal)

        # If moving apart or resting, no impulse needed (vel already corrected by penetration push?)
        if v_rel_normal >= -1e-4:  # Small tolerance for resting contact separation
            return

        # Use already updated world inverse inertia tensors
        inv_I_pin2_world = pin2.inv_inertia_world
        inv_I_pin1_world = pin1.inv_inertia_world

        # Impulse calculation terms
        term_pin2_ang = np.cross(inv_I_pin2_world @ np.cross(r_pin2_world, contact_normal), r_pin2_world)
        term_pin1_ang = np.cross(inv_I_pin1_world @ np.cross(r_pin1_world, contact_normal), r_pin1_world)
        ang_impulse_term = np.dot(term_pin2_ang + term_pin1_ang, contact_normal)

        impulse_denom = pin2.inv_mass + pin1.inv_mass + ang_impulse_term

        if abs(impulse_denom) < 1e-9:
            print("Warning: Impulse denominator near zero. Skipping impulse.")
            return

        impulse_j = -(1.0 + COLLISION_EPSILON) * v_rel_normal / impulse_denom
        impulse_vector = impulse_j * contact_normal

        # --- Apply Impulse ---
        pin2.vel += impulse_vector * pin2.inv_mass
        pin1.vel -= impulse_vector * pin1.inv_mass

        pin2.ang_vel += inv_I_pin2_world @ np.cross(r_pin2_world, impulse_vector)
        pin1.ang_vel -= inv_I_pin1_world @ np.cross(r_pin1_world, impulse_vector)

        # print(f"Collision Resolved: j={impulse_j:.3f}")

    else:
        if isinstance(obj1, Ball) and isinstance(obj2, Pin):
            ball = obj1
            pin = obj2
        elif isinstance(obj1, Pin) and isinstance(obj2, Ball):
            ball = obj2
            pin = obj1

            return # Skip Pin-Pin for now
        else:
            return # Unknown pair

        # --- Use object properties directly ---
        ball_pos, ball_vel, ball_q, ball_ang_vel = ball.pos, ball.vel, ball.quat, ball.ang_vel
        pin_pos, pin_vel, pin_q, pin_ang_vel = pin.pos, pin.vel, pin.quat, pin.ang_vel

        vec_pin_to_ball = ball_pos - pin_pos
        dist_centers_sq = np.dot(vec_pin_to_ball, vec_pin_to_ball)

        # Broad Phase
        bounding_radius_pin = np.sqrt(pin.radius**2 + (pin.height / 2)**2)
        min_dist_centers = ball.radius + bounding_radius_pin
        if dist_centers_sq > min_dist_centers**2:
            return # Too far apart

        # Detailed Check
        pin_z_axis_world = pin_q.rotate_vector([0, 0, 1])
        pin_y_axis_world = pin_q.rotate_vector([0, 1, 0])
        pin_x_axis_world = pin_q.rotate_vector([1, 0, 0])

        proj_len = np.dot(vec_pin_to_ball, pin_z_axis_world)
        closest_pt_on_axis = pin_pos + proj_len * pin_z_axis_world
        vec_axis_to_ball = ball_pos - closest_pt_on_axis
        dist_axis_to_ball_sq = np.dot(vec_axis_to_ball, vec_axis_to_ball)
        dist_axis_to_ball = np.sqrt(dist_axis_to_ball_sq) if dist_axis_to_ball_sq > 0 else 0
        is_within_height_range = abs(proj_len) <= pin.height / 2.0
        # Use slightly tolerance for range check robustness
        radial_thresh = pin.radius + ball.radius
        is_within_radial_range = dist_axis_to_ball < radial_thresh + 1e-5

        is_within_cap_height_range = pin.height / 2.0 < abs(proj_len) < pin.height / 2.0 + ball.radius + 1e-5
        is_radially_inside_cap_proj = dist_axis_to_ball < pin.radius + 1e-5 # Ball center needs project inside pin radius

        collided = False
        penetration_depth = 0.0
        contact_normal = np.zeros(3)
        contact_point_ball_local = np.zeros(3)
        contact_point_pin_local = np.zeros(3)
        contact_point_pin_world = np.zeros(3) # Store world point for calc

        # Check Wall Collision
        if is_within_height_range and is_within_radial_range:
            penetration_candidate = radial_thresh - dist_axis_to_ball
            if penetration_candidate > -1e-5: # Allow very slight overlap before trigger
                if dist_axis_to_ball > 1e-6:
                    contact_normal = vec_axis_to_ball / dist_axis_to_ball
                else: # Ball center is on the pin axis - push radially
                    contact_normal = pin_x_axis_world if abs(np.dot(vec_pin_to_ball, pin_x_axis_world)) > abs(np.dot(vec_pin_to_ball, pin_y_axis_world)) else pin_y_axis_world

                penetration_depth = max(0, penetration_candidate) # Ensure non-negative
                collided = True
                contact_point_ball_local = -contact_normal * ball.radius # Relative to ball center
                contact_point_pin_world = closest_pt_on_axis + contact_normal * pin.radius # On pin surface
                contact_point_pin_local = pin_q.conjugate().rotate_vector(contact_point_pin_world - pin_pos)
                # print(f"Wall Hit: depth={penetration_depth:.4f}")

        # Check Cap Collision (only if not already hit wall)
        # Use fixed cap contact point logic!
        if not collided and is_within_cap_height_range and is_radially_inside_cap_proj:
            penetration_candidate = ball.radius - (abs(proj_len) - pin.height / 2.0)
            if penetration_candidate > -1e-5:
                is_top_cap = proj_len > 0
                contact_normal = pin_z_axis_world if is_top_cap else -pin_z_axis_world
                penetration_depth = max(0, penetration_candidate)
                collided = True

                # --- Corrected Cap Contact Point ---
                cap_center_world = pin_pos + pin_z_axis_world * (pin.height / 2.0 * np.sign(proj_len))
                # vec_axis_to_ball is the vector from axis to ball center (in cap plane)
                contact_point_pin_world = cap_center_world + vec_axis_to_ball
                # --- End Correction ---

                contact_point_ball_local = -contact_normal * ball.radius
                contact_point_pin_local = pin_q.conjugate().rotate_vector(contact_point_pin_world - pin_pos)
                # print(f"Cap Hit: depth={penetration_depth:.4f}")

        if not collided or penetration_depth <= 0:
            return # No actual collision/penetration

        # --- Resolve Penetration (move objects apart based on mass) ---
        total_mass = ball.mass + pin.mass
        inv_total_mass = 1.0 / total_mass if total_mass > 1e-9 else 0
        move_fraction_ball = pin.mass * inv_total_mass
        move_fraction_pin = ball.mass * inv_total_mass

        # Correction factor slightly > 1 to ensure separation
        correction_factor = 1.01
        correction = contact_normal * penetration_depth * correction_factor
        ball.pos += correction * move_fraction_ball
        pin.pos -= correction * move_fraction_pin

        # --- Calculate Impulse (velocity change) ---
        r_ball_world = ball_q.rotate_vector(contact_point_ball_local) # Vector from ball CM to contact
        r_pin_world = pin_q.rotate_vector(contact_point_pin_local)   # Vector from pin CM to contact

        v_contact_ball = ball.vel + np.cross(ball.ang_vel, r_ball_world)
        v_contact_pin = pin.vel + np.cross(pin.ang_vel, r_pin_world)
        v_relative = v_contact_ball - v_contact_pin
        v_rel_normal = np.dot(v_relative, contact_normal)

        # If moving apart or resting, no impulse needed (vel already corrected by penetration push?)
        if v_rel_normal >= -1e-4: # Small tolerance for resting contact separation
            return

        # Use already updated world inverse inertia tensors
        inv_I_ball_world = ball.inv_inertia_world
        inv_I_pin_world = pin.inv_inertia_world

        # Impulse calculation terms
        term_ball_ang = np.cross(inv_I_ball_world @ np.cross(r_ball_world, contact_normal), r_ball_world)
        term_pin_ang = np.cross(inv_I_pin_world @ np.cross(r_pin_world, contact_normal), r_pin_world)
        ang_impulse_term = np.dot(term_ball_ang + term_pin_ang, contact_normal)

        impulse_denom = ball.inv_mass + pin.inv_mass + ang_impulse_term

        if abs(impulse_denom) < 1e-9:
            print("Warning: Impulse denominator near zero. Skipping impulse.")
            return

        impulse_j = -(1.0 + COLLISION_EPSILON) * v_rel_normal / impulse_denom
        impulse_vector = impulse_j * contact_normal

        # --- Apply Impulse ---
        ball.vel += impulse_vector * ball.inv_mass
        pin.vel -= impulse_vector * pin.inv_mass

        ball.ang_vel += inv_I_ball_world @ np.cross(r_ball_world, impulse_vector)
        pin.ang_vel -= inv_I_pin_world @ np.cross(r_pin_world, impulse_vector)

        # print(f"Collision Resolved: j={impulse_j:.3f}")


def get_lowest_point_on_cylinder (quat, pos, radius, height, num_samples=20):
    """
    Finds the lowest point on both the top and bottom circle of a cylinder
    in world space (accounting for tilt, rotation).
    """
    min_z = float('inf')
    lowest_point = None

    for cap in [-height / 2.0, height / 2.0]:  # bottom and top caps
        center_local = np.array([0, 0, cap])
        for i in range(num_samples):
            theta = 2 * np.pi * i / num_samples
            point_local = center_local + radius * np.array([np.cos(theta), np.sin(theta), 0])
            point_world = quat.rotate_vector(point_local) + pos

            if point_world[2] < min_z:
                min_z = point_world[2]
                lowest_point = point_world

    return lowest_point

# --- Floor Collision (Improved) ---
def resolve_floor_collisions(objects, dt):
    """Checks and resolves floor collisions for all objects in the list."""
    contact_normal = np.array([0.0, 0.0, 1.0])  # Floor normal points up

    for obj in objects:
        lowest_z = float('inf')
        contact_points_world = []

        # --- 1. Find Lowest Point(s) and Penetration ---
        if isinstance(obj, Ball):
            lowest_z_candidate = obj.pos[2] - obj.radius
            if lowest_z_candidate < lowest_z:
                lowest_z = lowest_z_candidate
            if lowest_z < 1e-5:  # Tolerance
                # Contact point is directly below the center for sphere
                contact_points_world.append(obj.pos - contact_normal * obj.radius)

        elif isinstance(obj, Pin):
            # Use the new helper function to find the lowest point on the pin
            lowest_point = get_lowest_point_on_cylinder(
                quat=obj.quat,
                pos=obj.pos,
                radius=obj.radius,
                height=obj.height,
                num_samples=20
            )

            lowest_z = lowest_point[2]
            if lowest_z < 1e-5:
                contact_points_world.append(lowest_point)

        if not contact_points_world or lowest_z >= -1e-5:  # No penetration
            continue

        # Average contact point if multiple vertices are penetrating (still valid for Ball)
        avg_contact_point_world = np.mean(contact_points_world, axis=0)
        penetration_depth = -lowest_z  # Since lowest_z is negative

        # Here you would resolve the collision
        # e.g., apply position correction or forces

        # --- 2. Resolve Penetration ---
        # Ensure we don't overcorrect if already slightly above due to previous steps
        actual_penetration = max(0, penetration_depth)
        correction = contact_normal * actual_penetration * 1.05 # Move slightly more
        obj.pos += correction
        # Clamp Z position just in case (prevents sinking through floor over time)
        if isinstance(obj, Ball):
             if obj.pos[2] < obj.radius: obj.pos[2] = obj.radius
        elif isinstance(obj, Pin):
            # Recheck lowest point after move - should be >= 0
             min_z_after = float('inf')
             pin_vertices_after = obj.get_vertices()
             for v_world in pin_vertices_after['bottom']:
                  min_z_after = min(min_z_after, v_world[2])
             if min_z_after < -1e-6: # Still somehow penetrating? Force slightly above.
                 obj.pos[2] += abs(min_z_after) + 1e-4
                 # print(f"Warning: Floor penetration clamp needed for {obj.id}")


        # --- 3. Calculate Normal Impulse (Bounce) ---
        r_world = avg_contact_point_world - obj.pos # Vector from CM to avg contact point

        v_contact = obj.vel + np.cross(obj.ang_vel, r_world)
        v_rel_normal = v_contact[2] # Z-component relative to static floor

        impulse_j_normal = 0.0 # Initialize

        # Only apply bounce impulse if approaching floor significantly
        if v_rel_normal < -1e-3:
            # Use object's world inverse inertia tensor
            inv_I_world = obj.inv_inertia_world
            term_ang = np.cross(inv_I_world @ np.cross(r_world, contact_normal), r_world)
            ang_impulse_term = np.dot(term_ang, contact_normal)
            impulse_denom = obj.inv_mass + ang_impulse_term

            if abs(impulse_denom) > 1e-9:
                impulse_j_normal = -(1.0 + FLOOR_RESTITUTION) * v_rel_normal / impulse_denom
                impulse_j_normal = max(0, impulse_j_normal) # Impulse must be non-negative (repulsive)
                impulse_vector_normal = impulse_j_normal * contact_normal

                # Apply normal impulse
                obj.vel += impulse_vector_normal * obj.inv_mass
                obj.ang_vel += inv_I_world @ np.cross(r_world, impulse_vector_normal)

        # --- 4. Calculate Friction Impulse (Rotation/Slowing) ---
        # Recalculate contact velocity *after* bounce impulse
        v_contact = obj.vel + np.cross(obj.ang_vel, r_world)
        v_tangential = v_contact.copy()
        v_tangential[2] = 0.0 # Project onto XY plane (tangent to floor)
        speed_tangential = np.linalg.norm(v_tangential)

        # Apply friction only if moving tangentially and there's contact force
        # Approximation: Significant normal impulse OR penetration occurred (resting/sliding)
        min_tangential_speed = 1e-3
        # Need a measure of normal force for friction: F_f <= mu * F_n
        # Rough approximation: Use normal impulse magnitude (if any) OR force needed to counteract gravity if penetrating/resting.
        # Let's use a simplified approach: Apply max friction if contact occurred and moving tangentially.
        if speed_tangential > min_tangential_speed and (impulse_j_normal > 1e-6 or actual_penetration > 1e-6) :
             friction_dir = -v_tangential / speed_tangential # Opposite to tangential velocity
             inv_I_world = obj.inv_inertia_world

             # Calculate impulse needed to stop tangential motion in this step
             term_ang_friction = np.cross(inv_I_world @ np.cross(r_world, friction_dir), r_world)
             ang_friction_term = np.dot(term_ang_friction, friction_dir)
             friction_denom = obj.inv_mass + ang_friction_term

             if abs(friction_denom) > 1e-9:
                  # Impulse magnitude needed to stop tangential motion
                  j_to_stop = speed_tangential / friction_denom

                  # Max friction impulse magnitude based on Coulomb friction (mu * N)
                  # Estimate Normal Force (N): Use gravity component + bounce impulse force
                  normal_force_magnitude = abs(obj.mass * GRAVITY[2]) # Force due to gravity
                  if impulse_j_normal > 0:
                      normal_force_magnitude += impulse_j_normal / dt # Add force from bounce impulse

                  j_friction_max = FLOOR_FRICTION_COEFF * normal_force_magnitude * dt # Max impulse over dt

                  # Actual friction impulse is the smaller of the two (cannot exceed max, cannot overcorrect velocity)
                  impulse_j_friction = min(j_to_stop, j_friction_max)
                  impulse_j_friction = max(0, impulse_j_friction) # Must be non-negative magnitude
                  impulse_vector_friction = impulse_j_friction * friction_dir

                  # Apply friction impulse
                  obj.vel += impulse_vector_friction * obj.inv_mass
                  obj.ang_vel += inv_I_world @ np.cross(r_world, impulse_vector_friction)



# --- Visualization ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal', adjustable='box') # Try to make axes scale equally

# Pre-calculate unit sphere mesh points
u_sph, v_sph = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_sph_unit = np.cos(u_sph) * np.sin(v_sph)
y_sph_unit = np.sin(u_sph) * np.sin(v_sph)
z_sph_unit = np.cos(v_sph)

# Pre-calculate unit cylinder mesh points
theta_cyl = np.linspace(0, 2 * np.pi, 20) # Reduced sides for performance
z_cyl_unit = np.linspace(-0.5, 0.5, 8)
theta_cyl_grid, z_cyl_grid_unit = np.meshgrid(theta_cyl, z_cyl_unit)
x_cyl_unit = np.cos(theta_cyl_grid)
y_cyl_unit = np.sin(theta_cyl_grid)
# Caps
r_cap = np.linspace(0, 1, 5)
theta_cap_grid, r_cap_grid = np.meshgrid(theta_cyl, r_cap) # Reuse theta_cyl
x_cap_unit = r_cap_grid * np.cos(theta_cap_grid)
y_cap_unit = r_cap_grid * np.sin(theta_cap_grid)

# Dictionary to store plot artists for updating efficiently
plot_artists = {}

def init_visualization():
    """Create plot artists for all objects."""
    global plot_artists
    plot_artists = {} # Clear existing artists

    # Draw Ground Plane Static
    plot_limit_x = 1.5 # Increased range for 10 pins
    plot_limit_y = 1.5
    plot_limit_z = 1.5
    ax.set_xlim(-1.5, plot_limit_x) # Start behind ball
    ax.set_ylim(-plot_limit_y, plot_limit_y)
    ax.set_zlim(0, plot_limit_z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    gx, gy = np.meshgrid(np.linspace(-1.5, plot_limit_x, 5), np.linspace(-plot_limit_y, plot_limit_y, 5))
    gz = np.zeros_like(gx)
    ax.plot_surface(gx, gy, gz, color='lightgrey', alpha=0.3, zorder=-1)

    # Create artists for each object
    for obj in sim_objects:
        if isinstance(obj, Ball):
            # Create ball surface plot
            surf = ax.plot_surface(x_sph_unit * obj.radius + obj.pos[0],
                                   y_sph_unit * obj.radius + obj.pos[1],
                                   z_sph_unit * obj.radius + obj.pos[2],
                                   color='darkblue', alpha=0.9)
            plot_artists[obj.id] = {'type': 'sphere', 'surface': surf}
        elif isinstance(obj, Pin):
            # Create pin surfaces (wall, top cap, bottom cap)
            wall_surf = ax.plot_surface(np.zeros_like(x_cyl_unit), np.zeros_like(y_cyl_unit), np.zeros_like(z_cyl_grid_unit), color='red', alpha=0.7)
            top_cap_surf = ax.plot_surface(np.zeros_like(x_cap_unit), np.zeros_like(y_cap_unit), np.zeros_like(x_cap_unit), color='white', alpha=0.8)
            bottom_cap_surf = ax.plot_surface(np.zeros_like(x_cap_unit), np.zeros_like(y_cap_unit), np.zeros_like(x_cap_unit), color='darkred', alpha=0.8)
            plot_artists[obj.id] = {'type': 'cylinder', 'wall': wall_surf, 'top': top_cap_surf, 'bottom': bottom_cap_surf}

    return list(plot_artists.keys()) # Return IDs perhaps? Or all artists flatten?


def update_visualization(frame):
    """Updates the positions and orientations of plot artists."""
    ax.set_title(f'Bowling Simulation (Frame {frame}) DT={DT:.3f}')

    artists_to_return = []

    for obj in sim_objects:
        obj_artists = plot_artists.get(obj.id)
        if not obj_artists: continue

        rot_matrix = obj.quat.to_rotation_matrix()

        if obj_artists['type'] == 'sphere':
            surf = obj_artists['surface']
            # Calculate new coordinates
            x_new = x_sph_unit * obj.radius + obj.pos[0]
            y_new = y_sph_unit * obj.radius + obj.pos[1]
            z_new = z_sph_unit * obj.radius + obj.pos[2]
            # Update surface data - requires segments
            verts = [list(zip(x_new.flatten(), y_new.flatten(), z_new.flatten()))]
            # Recreate the Poly3DCollection - inefficient but often necessary
            surf.remove()
            new_surf = ax.plot_surface(x_new, y_new, z_new, color='darkblue', alpha=0.9)
            plot_artists[obj.id]['surface'] = new_surf # Store new artist
            artists_to_return.append(new_surf)


        elif obj_artists['type'] == 'cylinder':
            # Transform unit cylinder points to world space
            # Wall
            cyl_points_local = np.vstack([
                (obj.radius * x_cyl_unit).flatten(),
                (obj.radius * y_cyl_unit).flatten(),
                (obj.height * z_cyl_grid_unit).flatten()
            ])
            cyl_points_world = rot_matrix @ cyl_points_local + obj.pos[:, np.newaxis]
            x_cyl = cyl_points_world[0, :].reshape(x_cyl_unit.shape)
            y_cyl = cyl_points_world[1, :].reshape(y_cyl_unit.shape)
            z_cyl = cyl_points_world[2, :].reshape(z_cyl_grid_unit.shape)

            # Top Cap
            cap_top_points_local = np.vstack([
                (obj.radius * x_cap_unit).flatten(),
                (obj.radius * y_cap_unit).flatten(),
                np.full(x_cap_unit.size, obj.height * 0.5)
            ])
            cap_top_points_world = rot_matrix @ cap_top_points_local + obj.pos[:, np.newaxis]
            x_cap_top = cap_top_points_world[0, :].reshape(x_cap_unit.shape)
            y_cap_top = cap_top_points_world[1, :].reshape(y_cap_unit.shape)
            z_cap_top = cap_top_points_world[2, :].reshape(x_cap_unit.shape)

            # Bottom Cap
            cap_bottom_points_local = np.vstack([
                (obj.radius * x_cap_unit).flatten(),
                (obj.radius * y_cap_unit).flatten(),
                np.full(x_cap_unit.size, -obj.height * 0.5)
            ])
            cap_bottom_points_world = rot_matrix @ cap_bottom_points_local + obj.pos[:, np.newaxis]
            x_cap_bottom = cap_bottom_points_world[0, :].reshape(x_cap_unit.shape)
            y_cap_bottom = cap_bottom_points_world[1, :].reshape(y_cap_unit.shape)
            z_cap_bottom = cap_bottom_points_world[2, :].reshape(x_cap_unit.shape)

            # --- Update surfaces (remove and recreate) ---
            obj_artists['wall'].remove()
            obj_artists['top'].remove()
            obj_artists['bottom'].remove()

            new_wall = ax.plot_surface(x_cyl, y_cyl, z_cyl, color='red', alpha=0.7, linewidth=0.5, edgecolors='k')
            new_top = ax.plot_surface(x_cap_top, y_cap_top, z_cap_top, color='white', alpha=0.8)
            new_bottom = ax.plot_surface(x_cap_bottom, y_cap_bottom, z_cap_bottom, color='darkred', alpha=0.8)

            plot_artists[obj.id]['wall'] = new_wall
            plot_artists[obj.id]['top'] = new_top
            plot_artists[obj.id]['bottom'] = new_bottom
            artists_to_return.extend([new_wall, new_top, new_bottom])

    # Need to return all artists that might change - potentially complex
    # Returning the axes object is simpler for blit=False
    # return artists_to_return # If trying blit=True (might not work well)
    return ax, # Simpler approach for blit=False

# --- Pause/Play Controls ---
def pause(event):
    global is_paused
    is_paused = True
def play(event):
    global is_paused
    is_paused = False
ax_pause = plt.axes([0.7, 0.8, 0.1, 0.05])
ax_play = plt.axes([0.81, 0.8, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause')
btn_play = Button(ax_play, 'Play')
btn_pause.on_clicked(pause)
btn_play.on_clicked(play)


# --- Main Simulation Loop (Update Function for Animation) ---
def update(frame):
    """Updates the animation frame"""
    # Declare that we are using the global variables sim_objects and is_paused
    global sim_objects, is_paused, ani # Add ani here as well for error handling

    if is_paused:
        # Need to return the artists for the animation to stay visible
        # When blit=False, returning the axes object is usually sufficient and safer.
        return ax,

    try:
        # --- Physics Steps ---
        # 1. Integrate motion for all objects
        for obj in sim_objects:
            rk4_step(obj, DT)

        # --- Collision Handling ---
        # 2. Check and resolve floor collisions first (stable base)
        resolve_floor_collisions(sim_objects, DT) # Pass the list

        # 3. Check and resolve object-object collisions
        for i in range(len(sim_objects)):
            for j in range(i + 1, len(sim_objects)):
                obj1 = sim_objects[i]
                obj2 = sim_objects[j]
                check_and_resolve_collision(obj1, obj2) # Modifies objects in place

        # --- Update Visualization ---
        return update_visualization(frame) # Call separate viz update

    except Exception as e:
        print(f"Error in frame {frame}:")
        traceback.print_exc()
        # Stop animation on error
        if ani and hasattr(ani, 'event_source') and ani.event_source:
            # Check if event source exists before trying to stop
             try:
                 ani.event_source.stop()
                 print("Animation stopped due to error.")
             except AttributeError:
                 print("Could not stop animation event source.")
        is_paused = True # Stop further updates by pausing
        return ax, # Return something to prevent further animation errors


# --- Run Animation ---
# Initialize visualization first to create artists
init_visualization()

ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES,
                              interval=max(1, int(DT * 1000)),
                              blit=False, # Blit=True is hard with 3D surface updates
                              repeat=False)

plt.show()
print("Animation finished or window closed.")