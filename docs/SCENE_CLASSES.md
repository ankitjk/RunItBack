# Scene classes - labelling definitions

The definitions of the scene classifier's classes, used when labelling with
`tools/scene_labeler.py`. This is the master copy; `runitback-labeler` keeps a
copy for its LLM prompt. Change it here first.

Written 2026-11-24 from the project owner's labelling conventions. Labels made
before then (`XUkRx6CpL04`) may not follow every rule exactly - in particular,
zoomed finishing blows were labelled as the state before the zoom. Its GAME!
screens were relabelled from `non_gameplay` to `respawning` on 2026-11-25.

**2026-11-26: three classes added** (`pre_game`, `game_end`, `finishing_blow`;
14 in all). Not yet supported by the label tool, training or inference, and
`XUkRx6CpL04` is not yet relabelled: its entrances are still `non_gameplay`,
its GAME! screens `respawning`, and its finishing blows the state before the zoom.

## Terms

- **Within the stage**: within the horizontal bounds of the stage and its
  platforms, and above the stage.
- **Offstage**: outside the stage's horizontal bounds (off either side), **or
  below the stage**, even between its edges (e.g. recovering from underneath).
- **Grounded**: walking, running, standing, shielding, doing a grounded move,
  or in jumpsquat (the start-up of a jump), on the stage or a platform.

## How a frame is classified

1. If the frame is not gameplay, it is `pre_game`, `game_end` or
   `non_gameplay`. If it is a finishing-blow zoom, it is `finishing_blow`. Stop.
   (See "Whole-screen classes".)
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
**Pauses are non_gameplay.** The entrances and countdown before GO! are
`pre_game`, and the GAME!/TIME! screen is `game_end` (see below).

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

Cue: until the respawn platform appears, one or both players have **no damage
percentage** shown on their HUD. If either player is respawning, the frame is
`respawning`.

The KO flash can happen on any stock; it shows that a player **was** knocked
out. Compare `finishing_blow`, which shows that a final-stock KO is **likely**.

## Whole-screen classes (added 2026-11-26)

These describe the whole screen, not the players. They are checked first and
override every player state.

**pre_game** - From the first frame the stage is shown after the loading
screen (the characters' entrances) through the countdown, until GO! appears.
The first frame showing GO! is gameplay: the players can act from that frame.

**game_end** - The **GAME!** screen, or the **TIME!** screen when the clock
runs out, from the first frame it appears until the white flash that wipes to
the results screen; `non_gameplay` starts at the flash's first frame. (The
GAME! screen was `respawning` from 2026-11-25 to 2026-11-26.)

**finishing_blow** - The **red flash** the game shows when it calculates that
a hit is likely to knock out a player on their **final stock**. From the first
frame the red flash starts until it is completely gone; the zoom and freeze
frame fall around the middle of it. It usually, but not always, ends in a KO;
label what the frame shows, not the outcome. It is not `respawning`: that flash means a KO happened, on any stock.

Order of checks for a frame: `non_gameplay` / `pre_game` / `game_end` ->
`finishing_blow` -> the player states, ranked as above.
