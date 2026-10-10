#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KEY = "task-0000_episode_00002530-300:311"

CAPTURE = Path(
    "diagnostics/results/"
    "pointworld_attention_interaction_overlay/"
    "interaction_attention_capture.npz"
)

TRAJ = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    + KEY.replace(":", "_")
    + ".npz"
)

INVENTORY = Path(
    "diagnostics/results/"
    "behavior_object_distance_motion_full578_20260810_114615/"
    "per_object.csv"
)

OUTDIR = CAPTURE.parent


# ============================================================
# Load attention
# ============================================================

with np.load(
    CAPTURE,
    allow_pickle=True,
) as z:

    coord = z["coord"]
    is_robot = z["is_robot"].astype(bool)

    scene_to_robot = (
        z["scene_to_robot"]
    )

    candidate = (
        z["candidate_robot_fraction"]
    )

    incoming = (
        z["incoming_from_robot"]
    )

    T = int(z["T"])
    Nr = int(z["Nr"])

    robot_exists = (
        z["robot_exists"]
        .astype(bool)
    )


scene_xyz = coord[
    ~is_robot
]

robot_xyz = coord[
    is_robot
]


# ============================================================
# Load existing validated object IDs / GT geometry.
# ============================================================

with np.load(
    TRAJ,
    allow_pickle=True,
) as z:

    obj = np.asarray(
        z["scene_object_id"]
    )

    gt = np.asarray(
        z["gt_scene_flows"]
    )


while obj.ndim > 2 and obj.shape[0] == 1:
    obj = obj[0]

while gt.ndim > 3 and gt.shape[0] == 1:
    gt = gt[0]


if obj.ndim == 2:
    obj0 = obj[0]
elif obj.ndim == 1:
    obj0 = obj
else:
    raise RuntimeError(
        f"unexpected object-id shape: {obj.shape}"
    )


assert gt.ndim == 3

gt0 = gt[0]

print(
    "attention scene xyz shape =",
    scene_xyz.shape,
)

print(
    "stored gt0 shape =",
    gt0.shape,
)

# No tolerance is introduced.
# Either the exact stored point order/coordinates match,
# or we stop rather than attach wrong labels.
exact_alignment = np.array_equal(
    scene_xyz,
    gt0,
)

print(
    "EXACT_SCENE_COORD_ALIGNMENT =",
    exact_alignment,
)

if not exact_alignment:

    if scene_xyz.shape == gt0.shape:

        print(
            "max absolute coordinate difference =",
            float(
                np.max(
                    np.abs(
                        scene_xyz
                        - gt0
                    )
                )
            ),
        )

    raise RuntimeError(
        "Scene/object alignment is not exact. "
        "Stopping before visualization."
    )


assert len(obj0) == len(scene_xyz)


# ============================================================
# Identify radio from existing inventory, not hard-coded ID.
# ============================================================

inv = pd.read_csv(
    INVENTORY
)

sample_inv = inv[
    inv["key"] == KEY
].copy()

radio_rows = sample_inv[
    sample_inv[
        "object_name"
    ].str.contains(
        "__radio_",
        regex=False,
    )
]

assert len(radio_rows) == 1

radio_id = int(
    radio_rows.iloc[0][
        "object_id"
    ]
)

radio_name = str(
    radio_rows.iloc[0][
        "object_name"
    ]
)

radio_mask = (
    obj0 == radio_id
)

print(
    "radio object_id =",
    radio_id,
)

print(
    "radio name =",
    radio_name,
)

print(
    "radio points =",
    int(radio_mask.sum()),
)


# ============================================================
# Robot/action trajectory.
# Current sample previously showed 11*152 = 1672 valid tokens.
# Still derive from actual robot_exists.
# ============================================================

flat_time = np.repeat(
    np.arange(T),
    Nr,
)

valid_time = flat_time[
    robot_exists.reshape(-1)
]

assert len(valid_time) == len(
    robot_xyz
)

robot_centroid = []

for t in range(T):

    pts = robot_xyz[
        valid_time == t
    ]

    assert len(pts) > 0

    robot_centroid.append(
        pts.mean(axis=0)
    )

robot_centroid = np.asarray(
    robot_centroid
)


# ============================================================
# Helper
# ============================================================

def base_plot(title):

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Entire observed scene as context.
    ax.scatter(
        scene_xyz[:, 0],
        scene_xyz[:, 1],
        scene_xyz[:, 2],
        s=1,
        alpha=0.05,
    )

    # GT radio points.
    ax.scatter(
        scene_xyz[
            radio_mask, 0
        ],
        scene_xyz[
            radio_mask, 1
        ],
        scene_xyz[
            radio_mask, 2
        ],
        s=18,
        marker="x",
        label="GT radio",
    )

    # All robot/action tokens.
    ax.scatter(
        robot_xyz[:, 0],
        robot_xyz[:, 1],
        robot_xyz[:, 2],
        s=3,
        alpha=0.15,
        label="robot/action tokens",
    )

    # Robot trajectory centroid through future time.
    ax.plot(
        robot_centroid[:, 0],
        robot_centroid[:, 1],
        robot_centroid[:, 2],
        marker="o",
        linewidth=2,
        label="robot trajectory centroid",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    return fig, ax


# ============================================================
# Head 6:
# scene QUERY -> robot/action KEY
# Show only scene queries whose serialized patch actually
# contains robot candidates. This is a structural condition,
# not an attention-magnitude threshold.
# ============================================================

h = 6

scene_mass_h6 = (
    scene_to_robot[
        ~is_robot,
        h,
    ]
)

scene_candidate = (
    candidate[
        ~is_robot
    ]
)

candidate_mask = (
    scene_candidate > 0
)

fig, ax = base_plot(
    "Head 6: Scene Query → Robot/Action Attention"
)

sc = ax.scatter(
    scene_xyz[
        candidate_mask, 0
    ],
    scene_xyz[
        candidate_mask, 1
    ],
    scene_xyz[
        candidate_mask, 2
    ],
    c=scene_mass_h6[
        candidate_mask
    ],
    s=10,
)

fig.colorbar(
    sc,
    ax=ax,
    shrink=0.7,
    label="Head 6 robot/action attention mass",
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTDIR
    / "head6_scene_to_robot_with_radio_and_trajectory.png",
    dpi=220,
)

plt.close(fig)


# Candidate-composition corrected Head 6.
h6_delta = (
    scene_mass_h6
    - scene_candidate
)

fig, ax = base_plot(
    "Head 6: Attention − Robot Candidate Fraction"
)

sc = ax.scatter(
    scene_xyz[
        candidate_mask, 0
    ],
    scene_xyz[
        candidate_mask, 1
    ],
    scene_xyz[
        candidate_mask, 2
    ],
    c=h6_delta[
        candidate_mask
    ],
    s=10,
)

fig.colorbar(
    sc,
    ax=ax,
    shrink=0.7,
    label="Attention minus candidate fraction",
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTDIR
    / "head6_minus_candidate_with_radio_and_trajectory.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# Heads 0 / 3 / 7:
# REAL robot/action queries -> WHICH scene keys?
#
# incoming[:,h] is the mean attention contribution from all
# real robot queries to each original token.
# ============================================================

for h in [0, 3, 7]:

    scene_incoming = (
        incoming[
            ~is_robot,
            h,
        ]
    )

    fig, ax = base_plot(
        f"Head {h}: Robot/Action Query → Scene Attention"
    )

    sc = ax.scatter(
        scene_xyz[:, 0],
        scene_xyz[:, 1],
        scene_xyz[:, 2],
        c=scene_incoming,
        s=6,
    )

    fig.colorbar(
        sc,
        ax=ax,
        shrink=0.7,
        label=(
            "Mean incoming attention "
            "from robot/action queries"
        ),
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTDIR
        / (
            f"head{h}_robot_to_scene_"
            "with_radio_and_trajectory.png"
        ),
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# Object-level quantitative ranking.
# This directly answers:
# "Which object receives robot/action-query attention?"
# ============================================================

name_map = {
    int(r.object_id):
        str(r.object_name)
    for _, r in sample_inv.iterrows()
}

rows = []

for oid in np.unique(obj0):

    mask = (
        obj0 == oid
    )

    row = {
        "object_id":
            int(oid),

        "object_name":
            name_map.get(
                int(oid),
                f"object_{int(oid)}",
            ),

        "n_points":
            int(mask.sum()),

        "head6_scene_to_robot_mean":
            float(
                scene_mass_h6[
                    mask
                ].mean()
            ),

        "head6_candidate_fraction_mean":
            float(
                scene_candidate[
                    mask
                ].mean()
            ),

        "head6_attention_minus_candidate":
            float(
                (
                    scene_mass_h6[
                        mask
                    ]
                    - scene_candidate[
                        mask
                    ]
                ).mean()
            ),
    }

    for h in [0, 3, 7]:

        # Sum is meaningful here:
        # it is the fraction of a robot query's attention
        # received by this object's points, averaged over
        # all real robot queries.
        row[
            f"head{h}_robot_to_object_attention_mass"
        ] = float(
            incoming[
                ~is_robot,
                h,
            ][mask].sum()
        )

    rows.append(row)


rank = pd.DataFrame(rows)

rank.to_csv(
    OUTDIR
    / "object_attention_ranking.csv",
    index=False,
)


print()
print("=" * 80)
print("HEAD 6: RADIO DESCRIPTIVE SUMMARY")
print("=" * 80)

print(
    "radio scene->robot attention mean =",
    float(
        scene_mass_h6[
            radio_mask
        ].mean()
    ),
)

print(
    "radio robot candidate fraction mean =",
    float(
        scene_candidate[
            radio_mask
        ].mean()
    ),
)

print(
    "radio attention-minus-candidate mean =",
    float(
        h6_delta[
            radio_mask
        ].mean()
    ),
)


for h in [0, 3, 7]:

    col = (
        f"head{h}_robot_to_object_attention_mass"
    )

    ordered = rank.sort_values(
        col,
        ascending=False,
    ).reset_index(drop=True)

    radio_rank = int(
        ordered.index[
            ordered["object_id"]
            == radio_id
        ][0]
    ) + 1

    radio_mass = float(
        ordered.loc[
            ordered["object_id"]
            == radio_id,
            col,
        ].iloc[0]
    )

    print()
    print("=" * 80)
    print(
        f"HEAD {h}: ROBOT QUERY -> OBJECT RANKING"
    )
    print("=" * 80)

    print(
        ordered[
            [
                "object_id",
                "object_name",
                "n_points",
                col,
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        "radio rank =",
        radio_rank,
        "/",
        len(ordered),
    )

    print(
        "radio attention mass =",
        radio_mass,
    )


print()
print(
    "SAVED =",
    OUTDIR,
)

print("NO NEW MODEL INTERVENTION = True")
print("OFFICIAL_CODE_MODIFIED = False")
