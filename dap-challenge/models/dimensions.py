"""Axes: B=batch, L=history, I=camera inputs, N=B*L*I, C=RGB channels, H/W=image size, D=features."""

I_INPUTS = 2  # Fixed order: arm camera, then Mac camera.
C_RGB = 3
A = 6  # Position/target channels: five arm joints plus the gripper.
D_IMG = 128  # Image embedding size.
D_POS = 32  # Position embedding size.
D_OBS = I_INPUTS * D_IMG + D_POS  # Combined features for one observation.
D_FUSION = 128  # Features supplied to the Gaussian heads.
