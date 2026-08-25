#pragma once

#include <cstddef>
#include <span>

#include "loader/frame_type.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

// Sampling weight for frames[index].
//
// Takes the whole game rather than one frame because the temporal
// curvature term needs the neighbouring plies. Callers already hold the
// full ordered frame sequence -- chunks are one game, in ply order, and
// stay that way until shuffling_frame_sampler -- so this costs no copying
// and no extra residency.
float ComputePositionSamplingWeight(std::span<const FrameType> frames,
                                    size_t index,
                                    const PositionSamplingConfig& config);

// Absolute temporal curvature at frames[index]: how far this position's
// evaluation deviates from the midpoint of its neighbours. Returns false
// for the first and last ply of a game, which have no two neighbours.
bool ComputeTemporalCurvature(std::span<const FrameType> frames, size_t index,
                              float* out_curvature);

// Material balance in pawns, from the perspective of the side to move.
// Read from the current-position plane bitboards, so it needs no board
// reconstruction.
float ComputeMaterialBalance(const FrameType& frame);

// Strength of the sacrifice signal at frames[index]: non-zero when this
// position is a material sacrifice, or falls within the decayed forward
// window of a recent one. Returns false when no sacrifice is in range.
bool ComputeSacrificeSignal(std::span<const FrameType> frames, size_t index,
                            const PositionSamplingConfig& config,
                            float* out_signal);

// Whether the position matches the configured material-configuration
// filter (total material at or below endgame_material_max, and queens
// present if endgame_require_queens).
bool MatchesEndgameFilter(const FrameType& frame,
                          const PositionSamplingConfig& config);

}  // namespace training
}  // namespace lczero
