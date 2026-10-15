"""Fixed dimensions of the approved actor; symbolic axes used in annotations.

B: batch size. L: total observation-history length. N = B * L.
C_in/C_out: input/output image channels for the operation in question.
H_in/W_in and H_out/W_out: input/output image height and width.
At the policy boundary C_in = C_RGB. Within CNN blocks the channels vary.
D_flat = L * D_OBS: concatenated features across the complete history.
These names document tensor axes; jaxtyping annotations alone do not enforce
their values. The network's explicit checks validate fixed input dimensions.
"""

C_RGB = 3
A = 6  # Position/target channels: five arm joints plus the gripper.
D_IMG = 128  # Image embedding size.
D_POS = 32  # Position embedding size.
D_OBS = D_IMG + D_POS  # Combined features for one observation.
D_FUSION = 128  # Features supplied to the Gaussian heads.
