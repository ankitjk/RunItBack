# Scene classes - labelling definitions

The definitions of the scene classifier's classes, used when labelling with
`tools/scene_labeler.py`. This is the master copy; `runitback-labeler` keeps a
copy for its LLM prompt. Change it here first.

Written 2026-11-24 from the project owner's labelling conventions. Labels made
before then (`XUkRx6CpL04`) may not follow every rule exactly - in particular,
zoomed finishing blows were labelled as the state before the zoom.

## Terms

- **Within the stage**: within the horizontal bounds of the stage and its
  platforms, and above the stage.
- **Offstage**: outside the stage's horizontal bounds (off either side), **or
  below the stage**, even between its edges (e.g. recovering from underneath).
- **Grounded**: walking, running, standing, shielding, doing a grounded move,
  or in jumpsquat (the start-up of a jump), on the stage or a platform.

## How a frame is classified

1. If the frame is not gameplay, it is `non_gameplay`. Stop.
2. Decide each player's state: **respawning, offstage, on_ledge,
   knocked_down, airborne, grounded**.
3. Take the higher-ranked of the two:
   **respawning > offstage > on_ledge > knocked_down > airborne > grounded**
4. If both players share that state, the class is `both_<state>`; otherwise
   `one_<state>`. `respawning` is one class whether one or both players are respawning.

Examples: one offstage and one on the ledge -> `one_offstage`. One under the
stage and one above it -> `one_offstage`. Both under the stage ->
`both_offstage`. One on the ledge and one knocked down -> `one_on_ledge`.

## Classes

**non_gameplay** - Loading screens, character select, stage select, results
screens, and anything else that isn't actual gameplay between the two players.
**Pauses are non_gameplay.**

**both_grounded** - Both players are grounded, within the stage.

**one_airborne** - One player is grounded within the stage. The other is in the
air within the stage.

**both_airborne** - Both players are in the air within the stage.

**one_offstage** - One player is offstage (off either side, or below the
stage). The other is anywhere else: within the stage (grounded or in the air),
or on the ledge.

**both_offstage** - Both players are offstage: off the sides, below the stage,
or one of each.

**one_on_ledge** - One player is hanging on either ledge (just outside the
stage's bounds). The other is within the stage: grounded, knocked down, or in
the air.

**both_on_ledge** - Both players are hanging on a ledge. If both are on the
*same* ledge it lasts only a moment: the player who grabbed it later is pushed off.

**one_knocked_down** - One player is knocked down (lying down, bouncing slightly
off the ground) or in a tech animation (teching to avoid being knocked down;
shows a slight yellow flash). The other player is within the stage.

**both_knocked_down** - Both players are knocked down or teching, within the stage.

**respawning** - Any of:
- a flash from one edge of the screen showing a player was just knocked out
- a player knocked into the background (star KO), or large and close, slamming
  against the screen
- a player on the angel (respawn) platform
- the stock count shown in the centre of the screen as a player respawns
- **the GAME! screen**

Cue: until the respawn platform appears, one or both players have **no damage
percentage** shown on their HUD. If either player is respawning, the frame is
`respawning`.

## Zoomed-in finishing blows - planned 12th class

The zoom does not always end in a kill, so it is **not** `respawning`, and
"the state before the zoom" can't be learned by a model that sees one frame at
a time. Planned: a separate class, `finishing_blow`, for the zoomed, slowed-down
shot. Label what the frame shows; whether it killed is decided later, from the
frames after it (a respawn follows, or not).

Not yet supported: RunItBack's label tool, training script and inference know
only the 11 classes above.
