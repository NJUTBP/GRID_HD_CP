Trajectory parameters
  * time length of trajectory: 1,000 s
  * timestep size: 0.2 s
  * diameter of enclosure: 180 cm

Generated neuron information in activity_csv.zip
  * csv contains activities (firing rates), one neuron per line
  * grid cells:
      - grid scale 40 cm
      - basis vectors oriented at 0 and 60 deg
      - 2000 neurons
      - offset phases randomly chosen
  * direction cells:
      - 2000 neurons
      - orientations randomly chosen
  * conjunctive grid x direction cells:
      - 10000 neurons
      - offset phases and orientations randomly chosen
  * non-grid spatial cells:
      - spatial firing fields are the sum of 2 gaussians
      - uniformly random peak positions
      - standard deviation 40 cm  
      - 1000 neurons
  * random cells:
      - Random activity selected every 10 timesteps
      - Gaussian with mean 0 and standard deviation 0.5, truncated between 0 & 1
      - Value for each timestep determined via third-order interpolation
      - 1000 neurons

Offsets in offset*.csv
  * one neuron per line in correspondence to activity*.csv
  * grid cells: phase offset as a fraction of scale along each basis vector
  * direction cells: direction offset in radians
  * conjunctive cells: phase offsets followed by direction offset