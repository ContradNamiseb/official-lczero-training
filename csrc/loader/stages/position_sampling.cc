#include "loader/stages/position_sampling.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace lczero {
namespace training {

bool ComputeTemporalCurvature(std::span<const FrameType> frames, size_t index,
                              float* out_curvature) {
  if (index == 0 || index + 1 >= frames.size()) return false;

  const float previous = frames[index - 1].root_q;
  const float current = frames[index].root_q;
  const float next = frames[index + 1].root_q;
  if (std::isnan(previous) || std::isnan(current) || std::isnan(next)) {
    return false;
  }

  // root_q is from the side-to-move's perspective, so its sign flips every
  // ply. Folding the three plies onto one fixed perspective means
  // q_fixed[j] = root_q[j] * (-1)^j, and the curvature there is
  //
  //   q_fixed[i] - (q_fixed[i-1] + q_fixed[i+1]) / 2
  //
  // Writing s = (-1)^i, both neighbours carry (-1)^(i-1) = (-1)^(i+1) = -s,
  // so that expands to
  //
  //   s*q[i] - ((-s)*q[i-1] + (-s)*q[i+1])/2 = s*(q[i] + (q[i-1]+q[i+1])/2)
  //
  // and s drops out under the absolute value. Hence the PLUS below: it is
  // the alternating perspective folded in, not a sign error. Subtracting
  // here would instead measure "whose turn is it", which is ~1.0 for every
  // position in a won game and carries no information at all.
  *out_curvature =
      std::abs(current + 0.5f * (previous + next));
  return true;
}

float ComputeMaterialBalance(const FrameType& frame) {
  // Input planes are 8 history steps of 13 bitboards; the first 13 are the
  // current position, ordered P N B R Q K for us then for them.
  static constexpr int kPieceValue[6] = {1, 3, 3, 5, 9, 0};
  int ours = 0;
  int theirs = 0;
  for (int piece = 0; piece < 6; ++piece) {
    ours += kPieceValue[piece] * std::popcount(frame.planes[piece]);
    theirs += kPieceValue[piece] * std::popcount(frame.planes[6 + piece]);
  }
  return static_cast<float>(ours - theirs);
}

namespace {

// Material at frames[at], expressed from the point of view of whoever is
// to move at frames[from].
//
// ComputeMaterialBalance is side-to-move relative, so its sign flips every
// ply exactly as root_q does. Comparing two plies without this correction
// measures the alternation, not the material change.
float MaterialFromPerspectiveOf(std::span<const FrameType> frames, size_t at,
                                size_t from) {
  const float value = ComputeMaterialBalance(frames[at]);
  const bool same_side = ((at ^ from) & 1u) == 0u;
  return same_side ? value : -value;
}

// Net material the side to move at `at` gives up over the next
// `window` plies. Positive means material was lost.
float NetMaterialGivenUp(std::span<const FrameType> frames, size_t at,
                         size_t window) {
  const size_t end = std::min(at + window, frames.size() - 1);
  if (end <= at) return 0.0f;
  return ComputeMaterialBalance(frames[at]) -
         MaterialFromPerspectiveOf(frames, end, at);
}

}  // namespace

bool ComputeSacrificeSignal(std::span<const FrameType> frames, size_t index,
                            const PositionSamplingConfig& config,
                            float* out_signal) {
  if (frames.size() < 2) return false;

  const size_t window = config.sacrifice_window_plies();
  const size_t lookahead = config.sacrifice_lookahead_plies();
  const float threshold = config.sacrifice_threshold();
  const float decay = config.sacrifice_decay();

  // Walk back over the propagation window looking for the sacrifice this
  // position is the continuation of, and keep the strongest claim on it.
  // The sacrifice itself is the offset == 0 case, so a position is scored
  // by its own event and by any recent one, whichever is larger.
  const size_t earliest = index >= lookahead ? index - lookahead : 0;
  float best = 0.0f;
  for (size_t source = earliest; source <= index; ++source) {
    const float given_up = NetMaterialGivenUp(frames, source, window);
    if (given_up < threshold) continue;
    // Only the side that made the sacrifice is credited with it; for the
    // opponent this position is just an ordinary reply.
    if (((source ^ index) & 1u) != 0u) continue;
    const float decayed =
        given_up * std::pow(decay, static_cast<float>(index - source));
    best = std::max(best, decayed);
  }

  if (best <= 0.0f) return false;
  *out_signal = best;
  return true;
}

bool MatchesEndgameFilter(const FrameType& frame,
                          const PositionSamplingConfig& config) {
  static constexpr int kPieceValue[6] = {1, 3, 3, 5, 9, 0};
  int total = 0;
  int queens = 0;
  for (int piece = 0; piece < 6; ++piece) {
    const int count = std::popcount(frame.planes[piece]) +
                      std::popcount(frame.planes[6 + piece]);
    total += kPieceValue[piece] * count;
    if (piece == 4) queens = count;
  }
  if (static_cast<float>(total) > config.endgame_material_max()) return false;
  if (config.endgame_require_queens() && queens == 0) return false;
  return true;
}

float ComputePositionSamplingWeight(std::span<const FrameType> frames,
                                    size_t index,
                                    const PositionSamplingConfig& config) {
  const FrameType& frame = frames[index];

  const bool has_diff_focus = config.has_diff_focus_q_weight() ||
                              config.has_diff_focus_pol_scale();
  const bool has_temporal = config.has_temporal_curvature_weight();
  const bool has_sacrifice = config.has_material_sacrifice_weight();
  const bool has_endgame = config.has_endgame_focus_weight();
  if (!has_diff_focus && !has_temporal && !has_sacrifice && !has_endgame) {
    return config.default_weight();
  }

  // Accumulated as a weighted mean so each term contributes only when it
  // has usable inputs. The previous shape of this function early-returned
  // default_weight whenever orig_q was NaN, which on a PGN-converted
  // corpus is every position -- that would have made the curvature term
  // below unreachable on exactly the data it exists to serve.
  float numerator = 0.0f;
  float denominator = 0.0f;

  if (has_diff_focus && !std::isnan(frame.orig_q)) {
    const float diff_q = std::abs(frame.best_q - frame.orig_q);
    numerator += config.diff_focus_q_weight() * diff_q + frame.policy_kld;
    denominator += config.diff_focus_q_weight() + config.diff_focus_pol_scale();
  }

  if (has_temporal) {
    float curvature = 0.0f;
    if (ComputeTemporalCurvature(frames, index, &curvature)) {
      numerator += config.temporal_curvature_weight() * curvature;
      denominator += config.temporal_curvature_weight();
    }
  }

  if (has_sacrifice) {
    // Contributes its weight to the denominator whether or not a sacrifice
    // is in range, so that quiet positions are pulled toward zero rather
    // than merely being left out of the average. Omitting them from the
    // denominator would make a quiet position score the same as a
    // sacrificial one, which is the opposite of the intent.
    float signal = 0.0f;
    if (ComputeSacrificeSignal(frames, index, config, &signal)) {
      numerator += config.material_sacrifice_weight() * signal;
    }
    denominator += config.material_sacrifice_weight();
  }

  if (has_endgame) {
    // Same convention as the sacrifice term: always in the denominator, so
    // positions outside the target phase are pushed down rather than
    // merely abstaining.
    if (MatchesEndgameFilter(frame, config)) {
      numerator += config.endgame_focus_weight();
    }
    denominator += config.endgame_focus_weight();
  }

  // No term had usable inputs: the first/last ply of a game under a
  // temporal-only config, or a NaN orig_q under a diff_focus-only one.
  if (denominator <= 0.0f) return config.default_weight();

  const float total = numerator / denominator;
  return std::min(
      std::pow(total * config.diff_focus_alpha() + config.diff_focus_beta(),
               config.diff_focus_gamma()),
      config.diff_focus_tau());
}

}  // namespace training
}  // namespace lczero
