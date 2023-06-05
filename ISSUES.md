# List of issues

Because my development folder is a fork of the work by Victor Wåhlstrand Skärström, I cannot add issues directly on Github, so I will add them here.

## New development items

- Land hitting: when a vessel reaches land (defined as a `nan` in the grid of currents), the simulation stops. This is problematic in those instances when bad weather pushes a boat towards the coast. Since some of the trips we are trying to simulate take place close to the coast we want to introduce a system for the boat to stir away from the land when it reaches it. However, the way the code was set up checks whether land is hit after moving, which means that it's too late for a boat to stir once land has been reached. So I need to develop a **radar function** that allows a vessel to stir away from land when it sees it ahead.

## Open questions

- Maximal speed and angle in polar diagrams: the polar diagrams (`./configs/hjortspring_speeds*` or `./configs/hjortspring_leeway*`) used to determine the boat speed and effective direction have a maximal wind speed of 30 knots, yet often winds can be stronger (in case of storms). Currently, the model is set up in such a way that if the wind speed is larger than 30 knots, the boat will take the values of speed and leeway equivalent for 30 knots. Is this sensible?
