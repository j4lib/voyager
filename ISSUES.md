# List of issues

Because my development folder is a fork of the work by Victor Wåhlstrand Skärström, I cannot add issues directly on Github, so I will add them here.

## New development items

- Land hitting: when a vessel reaches land (defined as a `nan` in the grid of currents), the simulation stops. This is problematic in those instances when bad weather pushes a boat towards the coast. Since some of the trips we are trying to simulate take place close to the coast we want to introduce a system for the boat to stir away from the land when it reaches it. However, the way the code was set up checks whether land is hit after moving, which means that it's too late for a boat to stir once land has been reached. So I need to develop a **radar function** that allows a vessel to stir away from land when it sees it ahead.
    - The radar function was implemented. This algorithm works in 3 stages: first, the vessel detects whether there is land in a radius of 0.05° of lat/lon around its position (which is about 5-6 km away); then, if there is some land, the algorithm calculates the direction of the closest piece of land, and it stirs away from the land by a certain angle. With 90° it can be problematic, as for example I found some positions where the land comes out perpendicularly from the coast, and a stir of 90° pushes the boat to land anyway. I will play around with the angle of stirring, but I think it will be something like 100-120°, so that in addition to pushing the boat away from the coast, it pushes it back away from the land ahead.

## Open questions

- Maximal speed and angle in polar diagrams: the polar diagrams (`./configs/hjortspring_speeds*` or `./configs/hjortspring_leeway*`) used to determine the boat speed and effective direction have a maximal wind speed of 30 knots, yet often winds can be stronger (in case of storms). Currently, the model is set up in such a way that if the wind speed is larger than 30 knots, the boat will take the values of speed and leeway equivalent for 30 knots. Is this sensible?
