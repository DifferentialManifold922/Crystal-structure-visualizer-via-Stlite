import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from itertools import combinations

# ================= 1. Core Engine: Calculations =================

HCP_STR = "六角最密堆积 (HCP)"
PEROVSKITE_STR = "钙钛矿 (ABO3)"
SC_STR = "简单立方 (SC)"
BCC_STR = "体心立方 (BCC)"
FCC_STR = "面心立方 (FCC)"

LATTICE_TYPES = {
    SC_STR: [[0.0, 0.0, 0.0]],
    BCC_STR: [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    FCC_STR: [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
    PEROVSKITE_STR: [[0.0, 0.0, 0.0]],
    HCP_STR: [[0.0, 0.0, 0.0]],
}


def get_cartesian_lattice(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    v_a = np.array([a, 0, 0])
    v_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0, c**2 - cx**2 - cy**2))
    v_c = np.array([cx, cy, cz])
    return np.array([v_a, v_b, v_c]).T


def draw_box(M, fig, color="#000000", width=2):
    v_f = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    v_c = np.dot(v_f, M.T)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    tx, ty, tz = [], [], []
    for s, e in edges:
        tx.extend([v_c[s][0], v_c[e][0], None])
        ty.extend([v_c[s][1], v_c[e][1], None])
        tz.extend([v_c[s][2], v_c[e][2], None])
    fig.add_trace(
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="lines",
            line=dict(color=color, width=width),
            name="单胞边框",
            showlegend=False,
            hoverinfo="none",
        )
    )


def draw_hcp_hex_prism(M, fig, color="#000000", width=2):
    """Draw a more intuitive hexagonal-prism reference frame for HCP."""
    a_vec = M[:, 0]
    b_vec = M[:, 1]
    c_vec = M[:, 2]

    # Basal hexagon vertices (clockwise), formed by a/b basis vectors.
    bottom = [
        1.0 * a_vec + 0.0 * b_vec,
        1.0 * a_vec + 1.0 * b_vec,
        0.0 * a_vec + 1.0 * b_vec,
        -1.0 * a_vec + 0.0 * b_vec,
        -1.0 * a_vec - 1.0 * b_vec,
        0.0 * a_vec - 1.0 * b_vec,
    ]
    top = [p + c_vec for p in bottom]

    tx, ty, tz = [], [], []

    # 6 edges on the bottom face
    for i in range(6):
        s, e = bottom[i], bottom[(i + 1) % 6]
        tx.extend([s[0], e[0], None])
        ty.extend([s[1], e[1], None])
        tz.extend([s[2], e[2], None])

    # 6 edges on the top face
    for i in range(6):
        s, e = top[i], top[(i + 1) % 6]
        tx.extend([s[0], e[0], None])
        ty.extend([s[1], e[1], None])
        tz.extend([s[2], e[2], None])

    # 6 vertical edges
    for i in range(6):
        s, e = bottom[i], top[i]
        tx.extend([s[0], e[0], None])
        ty.extend([s[1], e[1], None])
        tz.extend([s[2], e[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="lines",
            line=dict(color=color, width=width),
            name="HCP 六角柱边框",
            showlegend=False,
            hoverinfo="none",
        )
    )


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def _arc_points(u, v, radius, n_points=24):
    u_hat = _unit(u)
    v_hat = _unit(v)
    if u_hat is None or v_hat is None:
        return None

    cos_t = np.clip(np.dot(u_hat, v_hat), -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-6:
        return None

    e1 = u_hat
    v_perp = v_hat - np.dot(v_hat, e1) * e1
    e2 = _unit(v_perp)
    if e2 is None:
        return None

    ts = np.linspace(0.0, theta, n_points)
    return np.array([radius * (np.cos(t) * e1 + np.sin(t) * e2) for t in ts])


def draw_lattice_labels_on_cell(fig, M, a, b, c, alpha, beta, gamma):
    """Place lattice annotations near the unit cell: edge lengths + angle arcs."""
    va, vb, vc = M[:, 0], M[:, 1], M[:, 2]

    # 1) Edge-length labels near the midpoints of the three basis vectors
    edge_points = np.array([0.55 * va, 0.55 * vb, 0.55 * vc])
    edge_text = [f"a={a:.2f} Å", f"b={b:.2f} Å", f"c={c:.2f} Å"]
    fig.add_trace(
        go.Scatter3d(
            x=edge_points[:, 0],
            y=edge_points[:, 1],
            z=edge_points[:, 2],
            mode="text",
            text=edge_text,
            textposition="top center",
            textfont=dict(size=12, color="#111"),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # 2) Three angle arcs and labels: α(b-c), β(a-c), γ(a-b)
    angle_specs = [
        ("α", alpha, vb, vc),
        ("β", beta, va, vc),
        ("γ", gamma, va, vb),
    ]
    for symbol, value, u_vec, v_vec in angle_specs:
        radius = 0.22 * min(np.linalg.norm(u_vec), np.linalg.norm(v_vec))
        arc = _arc_points(u_vec, v_vec, radius=radius, n_points=28)
        if arc is None:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=arc[:, 0],
                y=arc[:, 1],
                z=arc[:, 2],
                mode="lines",
                line=dict(color="#2c3e50", width=3),
                showlegend=False,
                hoverinfo="none",
            )
        )

        mid = arc[len(arc) // 2] * 1.12
        fig.add_trace(
            go.Scatter3d(
                x=[mid[0]],
                y=[mid[1]],
                z=[mid[2]],
                mode="text",
                text=[f"{symbol}={value:.1f}°"],
                textposition="top center",
                textfont=dict(size=11, color="#2c3e50"),
                showlegend=False,
                hoverinfo="none",
            )
        )


def draw_crystal_axes(fig, M, axis_colors, axis_width=5):
    """Draw crystal axes a/b/c and show labels at axis ends."""
    origin = np.zeros(3)
    axis_defs = [
        ("a 轴", M[:, 0], axis_colors.get("a", "#e74c3c")),
        ("b 轴", M[:, 1], axis_colors.get("b", "#27ae60")),
        ("c 轴", M[:, 2], axis_colors.get("c", "#2980b9")),
    ]

    for label, vec, color in axis_defs:
        end = vec
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=color, width=axis_width),
                showlegend=False,
                hoverinfo="none",
            )
        )

        text_pos = 1.08 * end
        fig.add_trace(
            go.Scatter3d(
                x=[text_pos[0]],
                y=[text_pos[1]],
                z=[text_pos[2]],
                mode="text",
                text=[label],
                textposition="top center",
                textfont=dict(size=13, color=color),
                showlegend=False,
                hoverinfo="none",
            )
        )


def reciprocal_basis(M):
    """Return reciprocal basis matrix B without 2pi, where g = B @ [h,k,l]."""
    return np.linalg.inv(M).T


def miller_plane_info(M, h, k, l):
    """Compute normal, spacing, and intercept points for the (hkl) plane."""
    if h == 0 and k == 0 and l == 0:
        return None

    B = reciprocal_basis(M)
    g = B @ np.array([h, k, l], dtype=float)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-12:
        return None

    n_hat = g / g_norm
    d_hkl = 1.0 / g_norm
    p0 = g / (g_norm**2)  # Closest point to origin satisfying g·p0 = 1

    def axis_intercept(idx, m):
        if m == 0:
            return None
        frac = np.zeros(3)
        frac[idx] = 1.0 / m
        return M @ frac

    return {
        "g": g,
        "n_hat": n_hat,
        "d_hkl": d_hkl,
        "p0": p0,
        "ia": axis_intercept(0, h),
        "ib": axis_intercept(1, k),
        "ic": axis_intercept(2, l),
    }


def draw_miller_plane(fig, p0, n_hat, span, color="#16a085", opacity=0.25):
    """Draw a translucent patch for the (hkl) plane."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n_hat, ref)
    t1 = t1 / max(np.linalg.norm(t1), 1e-12)
    t2 = np.cross(n_hat, t1)
    t2 = t2 / max(np.linalg.norm(t2), 1e-12)

    s = 0.45 * span
    p1 = p0 + s * (t1 + t2)
    p2 = p0 + s * (t1 - t2)
    p3 = p0 + s * (-t1 - t2)
    p4 = p0 + s * (-t1 + t2)
    pts = np.array([p1, p2, p3, p4])

    fig.add_trace(
        go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=color,
            opacity=opacity,
            flatshading=True,
            showlegend=False,
            hoverinfo="none",
        )
    )


def draw_direction_uvw(fig, M, u, v, w, color="#8e44ad", width=7):
    """Draw an arrow line for the [uvw] crystal direction."""
    vec = M @ np.array([u, v, w], dtype=float)
    if np.linalg.norm(vec) < 1e-12:
        return
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, vec[0]],
            y=[0.0, vec[1]],
            z=[0.0, vec[2]],
            mode="lines+text",
            line=dict(color=color, width=width),
            text=["", f"[ {u} {v} {w} ]"],
            textposition="top center",
            textfont=dict(size=12, color=color),
            showlegend=False,
            hoverinfo="none",
        )
    )


def draw_bonds(fig, atoms, max_dist, color="#555", width=2, opacity=0.7):
    """Draw bond lines based on a distance cutoff."""
    n = len(atoms)
    if n < 2 or n > 1200:
        return

    tx, ty, tz = [], [], []
    for i, j in combinations(range(n), 2):
        pi = atoms[i]["pos"]
        pj = atoms[j]["pos"]
        d = np.linalg.norm(pi - pj)
        if 1e-6 < d <= max_dist:
            tx.extend([pi[0], pj[0], None])
            ty.extend([pi[1], pj[1], None])
            tz.extend([pi[2], pj[2], None])

    if tx:
        fig.add_trace(
            go.Scatter3d(
                x=tx,
                y=ty,
                z=tz,
                mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                showlegend=False,
                hoverinfo="none",
            )
        )


def draw_polyhedra(fig, atoms, center_elem, ligand_elem, cutoff, color="#3498db", opacity=0.18):
    """Build center-ligand convex polyhedra (Mesh3d alphahull=0)."""
    centers = [a for a in atoms if str(a["elem"]) == center_elem]
    ligands = [a for a in atoms if str(a["elem"]) == ligand_elem]
    if not centers or not ligands:
        return

    for c in centers:
        nbr = []
        for l in ligands:
            d = np.linalg.norm(c["pos"] - l["pos"])
            if 1e-6 < d <= cutoff:
                nbr.append(l["pos"])
        if len(nbr) >= 4:
            pts = np.array(nbr)
            fig.add_trace(
                go.Mesh3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    alphahull=0,
                    color=color,
                    opacity=opacity,
                    flatshading=True,
                    showlegend=False,
                    hoverinfo="none",
                )
            )


def ws_vertices_wireframe(M, lattice_pts, neighbor_range=2, tol=1e-7):
    """Construct a standard Wigner-Seitz cell from perpendicular bisector planes.

    Method:
    1. Generate neighboring lattice vectors in real space.
    2. Sort by distance and keep nearest neighbors.
    3. For each neighbor vector r, build a plane through r/2 with normal r.
    4. Solve plane intersections (Voronoi vertices).
    5. Connect adjacent vertices.
    """
    # 1) Generate all neighbor lattice vectors and their distances
    vectors = []
    for i in range(-neighbor_range, neighbor_range + 1):
        for j in range(-neighbor_range, neighbor_range + 1):
            for k in range(-neighbor_range, neighbor_range + 1):
                cell = np.array([i, j, k], dtype=float)
                for lp in lattice_pts:
                    f = cell + np.array(lp, dtype=float)
                    if np.linalg.norm(f) < tol:
                        continue
                    r_cart = M @ f
                    dist = np.linalg.norm(r_cart)
                    vectors.append((dist, r_cart))
    
    if not vectors:
        return np.empty((0, 3)), []
    
    # 2) Sort by distance and keep the nearest 12 neighbors
    vectors.sort(key=lambda x: x[0])
    vecs = np.array([v[1] for v in vectors[:12]])
    
    # 3) Build WS planes: normal = neighbor vector, passing through r/2
    # Plane equation: n·x = d, where n = r and d = ||r||^2 / 2
    normals = vecs
    offsets = 0.5 * np.sum(vecs * vecs, axis=1)
    
    # 4) Solve Voronoi vertices (intersections of three planes)
    verts = []
    m = len(normals)
    for i, j, k in combinations(range(m), 3):
        A = np.vstack([normals[i], normals[j], normals[k]])
        if abs(np.linalg.det(A)) < 1e-10:
            continue
        b = np.array([offsets[i], offsets[j], offsets[k]])
        try:
            x = np.linalg.solve(A, b)
            # Check if the point is inside all plane-defined half-spaces
            if np.all(normals @ x <= offsets + 1e-5):
                verts.append(x)
        except:
            continue
    
    if not verts:
        return np.empty((0, 3)), []
    
    # 5) Deduplicate vertices
    vuniq = {}
    for v in verts:
        key = tuple(np.round(v, 5))
        if key not in vuniq:
            vuniq[key] = np.array(key)
    V = np.array(list(vuniq.values()))
    if len(V) < 4:
        return V, []
    
    # 6) Identify active planes for each vertex
    active = []
    for v in V:
        idx = set(np.where(np.abs(normals @ v - offsets) < 1e-4)[0].tolist())
        active.append(idx)
    
    # 7) Connect vertices: an edge shares at least two active planes
    edges = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if len(active[i].intersection(active[j])) >= 2:
                edges.append((i, j))
    
    return V, edges


# ================= 2. UI Layout =================

st.set_page_config(page_title="Crystal Visualizer", layout="wide")
st.title("Crystal Visualizer")

# --- A. Sidebar ---
st.sidebar.header("设置")
lattice_choice = st.sidebar.selectbox(
    "选择晶格", list(LATTICE_TYPES.keys()), index=0
)
model_mode = st.sidebar.selectbox(
    "可视化模式",
    ["晶格点阵+基元", "默认"],
    index=1,
    key="model_mode",
)
lattice_pts = LATTICE_TYPES[lattice_choice]
full_cell_mode = model_mode == "默认"

# Use full conventional-cell basis in default mode to avoid extra Bravais-offset overlap.
if full_cell_mode and lattice_choice in {SC_STR, BCC_STR, FCC_STR, HCP_STR, PEROVSKITE_STR}:
    lattice_pts = [[0.0, 0.0, 0.0]]

st.sidebar.markdown("---")
st.sidebar.subheader("重复")
rx = st.sidebar.slider("X 方向", -6, 6, (0, 0), key="rx")
ry = st.sidebar.slider("Y 方向", -6, 6, (0, 0), key="ry")
rz = st.sidebar.slider("Z 方向", -6, 6, (0, 0), key="rz")

st.sidebar.markdown("---")
st.sidebar.subheader("原子样式")
global_scale = st.sidebar.slider("原子大小缩放", 5, 60, 25)
atom_opacity = st.sidebar.slider("原子透明度", 0.0, 1.0, 0.8)
box_color = st.sidebar.color_picker("单胞框颜色", "#f8f9fa")
box_width = st.sidebar.slider("单胞框线宽", 1, 8, 2)
show_axes = st.sidebar.checkbox("显示晶轴 a,b,c", value=True)
axis_width = st.sidebar.slider("晶轴线宽", 1, 12, 6)
axis_color_a = st.sidebar.color_picker("a 轴颜色", "#e74c3c", key="axis_color_a")
axis_color_b = st.sidebar.color_picker("b 轴颜色", "#27ae60", key="axis_color_b")
axis_color_c = st.sidebar.color_picker("c 轴颜色", "#2980b9", key="axis_color_c")

st.sidebar.markdown("---")
st.sidebar.subheader("晶面与晶向")
h = st.sidebar.number_input("h", value=1, step=1, format="%d")
k = st.sidebar.number_input("k", value=1, step=1, format="%d")
l = st.sidebar.number_input("l", value=1, step=1, format="%d")
u = st.sidebar.number_input("u", value=1, step=1, format="%d")
v = st.sidebar.number_input("v", value=0, step=1, format="%d")
w = st.sidebar.number_input("w", value=0, step=1, format="%d")
show_hkl_plane = st.sidebar.checkbox("显示 (hkl) 晶面", value=False, key="show_hkl_plane")
show_uvw_dir = st.sidebar.checkbox("显示 [uvw] 晶向", value=False, key="show_uvw_dir")
plane_color = st.sidebar.color_picker("晶面颜色", "#16a085", key="plane_color")
plane_opacity = st.sidebar.slider("晶面透明度", 0.05, 0.9, 0.22, 0.01, key="plane_opacity")
uvw_color = st.sidebar.color_picker("晶向颜色", "#e74c3c", key="uvw_color")
uvw_width = st.sidebar.slider("晶向箭线宽度", 1, 10, 5, key="uvw_width")

st.sidebar.markdown("---")
st.sidebar.subheader("晶面切片")
enable_slice = st.sidebar.checkbox("高亮晶面上的原子", value=False, key="enable_slice")

st.sidebar.markdown("---")
st.sidebar.subheader("指定原子的 WS 元胞")
show_atom_ws = st.sidebar.checkbox("绘制指定原子的 WS 元胞", value=False, key="show_atom_ws")
if show_atom_ws:
    atom_ws_elem = st.sidebar.text_input("原子元素（留空则按全部原子）", value="", key="atom_ws_elem")
    atom_ws_index = st.sidebar.number_input("原子索引（从 0 开始）", value=0, step=1, key="atom_ws_index")
    atom_ws_neighbor_range = st.sidebar.slider("WS 搜索范围", 1, 3, 2, key="atom_ws_neighbor_range")
    atom_ws_color = st.sidebar.color_picker("原子 WS 元胞框颜色", "#e74c3c", key="atom_ws_color")
    atom_ws_width = st.sidebar.slider("原子 WS 元胞框线宽", 1, 6, 2, key="atom_ws_width")

param_display_mode = st.sidebar.selectbox(
    "晶格参数显示方式",
    ["不显示", "图例", "图内标注"],
    index=0,
    key="param_display_mode",
)

# --- B. Main panel: split tabs ---
tab_geo, tab_basis = st.tabs(["几何参数", "自定义基元"])

with tab_geo:
    st.info(
        "a、b、c 为晶胞三条晶轴长度 (Å); α 为 b-c 夹角, β 为 a-c 夹角, γ 为 a-b 夹角 (角度制)。"
    )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        default_a = 3.9 if lattice_choice == PEROVSKITE_STR else 4.0
        in_a = st.number_input("a (Å)", value=default_a)
        in_al = st.number_input("α (°)", value=90.0)
    with p_col2:
        if lattice_choice == HCP_STR:
            default_b = in_a
        elif lattice_choice == PEROVSKITE_STR:
            default_b = 3.9
        else:
            default_b = 4.0
        in_b = st.number_input("b (Å)", value=default_b)
        in_be = st.number_input("β (°)", value=90.0)
    with p_col3:
        if lattice_choice == HCP_STR:
            default_c = float(in_a * np.sqrt(8 / 3))
        elif lattice_choice == PEROVSKITE_STR:
            default_c = 3.9
        else:
            default_c = 4.0
        in_c = st.number_input("c (Å)", value=default_c)
        default_ga = 120.0 if lattice_choice == HCP_STR else 90.0
        in_ga = st.number_input("γ (°)", value=default_ga)

    lattice_M = get_cartesian_lattice(in_a, in_b, in_c, in_al, in_be, in_ga)

with tab_basis:
    if full_cell_mode:
        if lattice_choice == HCP_STR:
            # Classic 17-atom hexagonal prism model (top/bottom hexagons + centers + 3 middle atoms)
            default_basis = [
                # --- Bottom face (z=0): 1 center + 6 vertices ---
                {"元素": "Mg", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Mg", "x": 1.0, "y": 0.0, "z": 0.0},
                {"元素": "Mg", "x": 1.0, "y": 1.0, "z": 0.0},
                {"元素": "Mg", "x": 0.0, "y": 1.0, "z": 0.0},
                {"元素": "Mg", "x": -1.0, "y": 0.0, "z": 0.0},
                {"元素": "Mg", "x": -1.0, "y": -1.0, "z": 0.0},
                {"元素": "Mg", "x": 0.0, "y": -1.0, "z": 0.0},
                # --- Top face (z=1): 1 center + 6 vertices ---
                {"元素": "Mg", "x": 0.0, "y": 0.0, "z": 1.0},
                {"元素": "Mg", "x": 1.0, "y": 0.0, "z": 1.0},
                {"元素": "Mg", "x": 1.0, "y": 1.0, "z": 1.0},
                {"元素": "Mg", "x": 0.0, "y": 1.0, "z": 1.0},
                {"元素": "Mg", "x": -1.0, "y": 0.0, "z": 1.0},
                {"元素": "Mg", "x": -1.0, "y": -1.0, "z": 1.0},
                {"元素": "Mg", "x": 0.0, "y": -1.0, "z": 1.0},
                # --- Middle layer (z=0.5): 3 atoms ---
                {"元素": "Mg", "x": 2 / 3, "y": 1 / 3, "z": 0.5},
                {"元素": "Mg", "x": -1 / 3, "y": 1 / 3, "z": 0.5},
                {"元素": "Mg", "x": -1 / 3, "y": -2 / 3, "z": 0.5},
            ]
        elif lattice_choice == PEROVSKITE_STR:
            default_basis = [
                {"元素": "Ca", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Ca", "x": 1.0, "y": 0.0, "z": 0.0},
                {"元素": "Ca", "x": 0.0, "y": 1.0, "z": 0.0},
                {"元素": "Ca", "x": 1.0, "y": 1.0, "z": 0.0},
                {"元素": "Ca", "x": 0.0, "y": 0.0, "z": 1.0},
                {"元素": "Ca", "x": 1.0, "y": 0.0, "z": 1.0},
                {"元素": "Ca", "x": 0.0, "y": 1.0, "z": 1.0},
                {"元素": "Ca", "x": 1.0, "y": 1.0, "z": 1.0},
                {"元素": "Ti", "x": 0.5, "y": 0.5, "z": 0.5},
                {"元素": "O", "x": 0.5, "y": 0.5, "z": 0.0},
                {"元素": "O", "x": 0.5, "y": 0.5, "z": 1.0},
                {"元素": "O", "x": 0.5, "y": 0.0, "z": 0.5},
                {"元素": "O", "x": 0.5, "y": 1.0, "z": 0.5},
                {"元素": "O", "x": 0.0, "y": 0.5, "z": 0.5},
                {"元素": "O", "x": 1.0, "y": 0.5, "z": 0.5},
            ]
        elif lattice_choice == SC_STR:
            default_basis = [
                {"元素": "Atom", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Atom", "x": 1.0, "y": 0.0, "z": 0.0},
                {"元素": "Atom", "x": 0.0, "y": 1.0, "z": 0.0},
                {"元素": "Atom", "x": 1.0, "y": 1.0, "z": 0.0},
                {"元素": "Atom", "x": 0.0, "y": 0.0, "z": 1.0},
                {"元素": "Atom", "x": 1.0, "y": 0.0, "z": 1.0},
                {"元素": "Atom", "x": 0.0, "y": 1.0, "z": 1.0},
                {"元素": "Atom", "x": 1.0, "y": 1.0, "z": 1.0},
            ]
        elif lattice_choice == BCC_STR:
            default_basis = [
                {"元素": "Atom", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Atom", "x": 1.0, "y": 0.0, "z": 0.0},
                {"元素": "Atom", "x": 0.0, "y": 1.0, "z": 0.0},
                {"元素": "Atom", "x": 1.0, "y": 1.0, "z": 0.0},
                {"元素": "Atom", "x": 0.0, "y": 0.0, "z": 1.0},
                {"元素": "Atom", "x": 1.0, "y": 0.0, "z": 1.0},
                {"元素": "Atom", "x": 0.0, "y": 1.0, "z": 1.0},
                {"元素": "Atom", "x": 1.0, "y": 1.0, "z": 1.0},
                {"元素": "Atom", "x": 0.5, "y": 0.5, "z": 0.5},
            ]
        elif lattice_choice == FCC_STR:
            default_basis = [
                {"元素": "Atom", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Atom", "x": 1.0, "y": 0.0, "z": 0.0},
                {"元素": "Atom", "x": 0.0, "y": 1.0, "z": 0.0},
                {"元素": "Atom", "x": 1.0, "y": 1.0, "z": 0.0},
                {"元素": "Atom", "x": 0.0, "y": 0.0, "z": 1.0},
                {"元素": "Atom", "x": 1.0, "y": 0.0, "z": 1.0},
                {"元素": "Atom", "x": 0.0, "y": 1.0, "z": 1.0},
                {"元素": "Atom", "x": 1.0, "y": 1.0, "z": 1.0},
                {"元素": "Atom", "x": 0.5, "y": 0.5, "z": 0.0},
                {"元素": "Atom", "x": 0.5, "y": 0.5, "z": 1.0},
                {"元素": "Atom", "x": 0.5, "y": 0.0, "z": 0.5},
                {"元素": "Atom", "x": 0.5, "y": 1.0, "z": 0.5},
                {"元素": "Atom", "x": 0.0, "y": 0.5, "z": 0.5},
                {"元素": "Atom", "x": 1.0, "y": 0.5, "z": 0.5},
            ]
        else:
            default_basis = [{"元素": "Atom", "x": 0.0, "y": 0.0, "z": 0.0}]
    else:
        if lattice_choice == HCP_STR:
            default_basis = [
                {"元素": "Mg", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Mg", "x": 2 / 3, "y": 1 / 3, "z": 0.5},
            ]
        elif lattice_choice == PEROVSKITE_STR:
            default_basis = [
                {"元素": "Ca", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "Ti", "x": 0.5, "y": 0.5, "z": 0.5},
                {"元素": "O", "x": 0.5, "y": 0.5, "z": 0.0},
                {"元素": "O", "x": 0.5, "y": 0.0, "z": 0.5},
                {"元素": "O", "x": 0.0, "y": 0.5, "z": 0.5},
            ]
        elif lattice_choice == FCC_STR:
            # In lattice+basis mode, FCC defaults to a diamond-like basis demo.
            default_basis = [
                {"元素": "C", "x": 0.0, "y": 0.0, "z": 0.0},
                {"元素": "C", "x": 0.25, "y": 0.25, "z": 0.25},
            ]
        else:
            default_basis = [{"元素": "Atom", "x": 0.0, "y": 0.0, "z": 0.0}]

    basis_df = st.data_editor(
        pd.DataFrame(default_basis),
        num_rows="dynamic",
        use_container_width=True,
        key="basis_editor",
    )

# Custom atom colors
unique_elements = basis_df["元素"].unique()
user_styles = {}
with st.sidebar.expander("原子颜色设置", expanded=False):
    for i, elem in enumerate(unique_elements):
        if pd.isna(elem):
            continue
        c = st.color_picker(
            f"{elem} 颜色", ["#3498db", "#e74c3c", "#f1c40f"][i % 3], key=f"cp_{elem}"
        )
        s = st.slider(f"{elem} 大小", 0.5, 2.0, 1.0, key=f"sz_{elem}")
        user_styles[elem] = {"color": c, "size": s}

# ================= 3. Core Rendering =================

fig = go.Figure()
all_atoms = []

# Generate atom coordinates
for ix in range(rx[0], rx[1] + 1):
    for iy in range(ry[0], ry[1] + 1):
        for iz in range(rz[0], rz[1] + 1):
            offset = np.array([ix, iy, iz])
            for lp in lattice_pts:
                for _, row in basis_df.iterrows():
                    if pd.isna(row["元素"]):
                        continue
                    f_pos = (
                        offset + np.array(lp) + np.array([row["x"], row["y"], row["z"]])
                    )
                    c_pos = np.dot(lattice_M, f_pos)
                    all_atoms.append({"pos": c_pos, "elem": row["元素"]})

# Draw atoms grouped by element
for elem in unique_elements:
    if pd.isna(elem):
        continue
    group = [a for a in all_atoms if a["elem"] == elem]
    if not group:
        continue
    marker_size = global_scale * user_styles[elem]["size"]
    fig.add_trace(
        go.Scatter3d(
            x=[a["pos"][0] for a in group],
            y=[a["pos"][1] for a in group],
            z=[a["pos"][2] for a in group],
            mode="markers",
            name=elem,
            marker=dict(
                size=marker_size,
                color=user_styles[elem]["color"],
                opacity=atom_opacity,
                line=dict(width=1, color="white"),
            ),
        )
    )

# ===== Advanced features =====

# Compute (hkl) plane information when needed
if show_hkl_plane or enable_slice:
    plane_info = miller_plane_info(lattice_M, int(h), int(k), int(l))
else:
    plane_info = None

# Slice handling
display_atoms = list(all_atoms)
slice_atoms = []

if plane_info and enable_slice:
    # Find atoms within slice thickness (fixed 0.1 A)
    g = plane_info["g"]
    g_norm = np.linalg.norm(g)
    radius = 0.05  # Half of the fixed slice thickness
    for a in all_atoms:
        dist = abs(np.dot(g, a["pos"] - plane_info["p0"])) / g_norm
        if dist <= radius:
            slice_atoms.append(a)

# Draw / highlight sliced atoms
if slice_atoms and enable_slice:
    fig.add_trace(
        go.Scatter3d(
            x=[a["pos"][0] for a in slice_atoms],
            y=[a["pos"][1] for a in slice_atoms],
            z=[a["pos"][2] for a in slice_atoms],
            mode="markers",
            name="切片高亮",
            marker=dict(
                size=global_scale * user_styles.get(slice_atoms[0]["elem"], {"size": 8})["size"] * 1.3,
                color="yellow",
                opacity=0.6,
                line=dict(width=2, color="orange"),
            ),
            showlegend=True,
        )
    )

# Draw crystal plane
if show_hkl_plane and plane_info:
    draw_miller_plane(
        fig,
        plane_info["p0"],
        plane_info["n_hat"],
        np.linalg.norm(np.array([in_a, in_b, in_c])) * 1.5,
        color=plane_color,
        opacity=plane_opacity,
    )

# Draw crystal direction
if show_uvw_dir:
    draw_direction_uvw(
        fig,
        lattice_M,
        int(u),
        int(v),
        int(w),
        color=uvw_color,
        width=uvw_width,
    )


def atom_ws_vertices_wireframe(center_atom, all_atoms, M, neighbor_range=1, tol=1e-7):
    """Compute a Wigner-Seitz cell for a single atom.

    Neighbors are defined by perpendicular bisector planes from center-to-neighbor vectors.
    """
    center_pos = center_atom["pos"]
    
    # Build vectors from the center atom to other atoms (including periodic images)
    vectors = []
    a_vec, b_vec, c_vec = M[:, 0], M[:, 1], M[:, 2]
    for atom in all_atoms:
        for ix in range(-neighbor_range, neighbor_range + 1):
            for iy in range(-neighbor_range, neighbor_range + 1):
                for iz in range(-neighbor_range, neighbor_range + 1):
                    shift = ix * a_vec + iy * b_vec + iz * c_vec
                    r_vec = atom["pos"] + shift - center_pos
                    dist = np.linalg.norm(r_vec)
                    if dist < tol:
                        continue
                    vectors.append((dist, r_vec))
    
    if not vectors:
        return np.empty((0, 3)), []
    
    # Deduplicate and sort by distance; keep nearest-neighbor directions
    uniq = {}
    for d, v in vectors:
        key = tuple(np.round(v, 5))
        uniq[key] = (d, np.array(key))
    vectors = list(uniq.values())
    vectors.sort(key=lambda x: x[0])
    vecs = np.array([v[1] for v in vectors[:24]])
    
    # Build perpendicular-bisector planes: normal = neighbor-center vector, through midpoint
    # Plane equation: n·x = d, with n = r and d = n·(r/2 + center)
    normals = vecs
    offsets = 0.5 * np.sum(vecs * vecs, axis=1) + normals @ center_pos
    
    # Solve Voronoi vertices
    verts = []
    m = len(normals)
    for i, j, k in combinations(range(m), 3):
        A = np.vstack([normals[i], normals[j], normals[k]])
        if abs(np.linalg.det(A)) < 1e-10:
            continue
        b = np.array([offsets[i], offsets[j], offsets[k]])
        try:
            x = np.linalg.solve(A, b)
            if np.all(normals @ x <= offsets + 1e-5):
                verts.append(x)
        except:
            continue
    
    if not verts:
        return np.empty((0, 3)), []
    
    # Deduplicate
    vuniq = {}
    for v in verts:
        key = tuple(np.round(v, 5))
        if key not in vuniq:
            vuniq[key] = np.array(key)
    V = np.array(list(vuniq.values()))
    if len(V) < 4:
        return V, []
    
    # Vertices sharing active planes form edges
    active = []
    for v in V:
        idx = set(np.where(np.abs(normals @ v - offsets) < 1e-4)[0].tolist())
        active.append(idx)
    
    edges = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if len(active[i].intersection(active[j])) >= 2:
                edges.append((i, j))
    
    return V, edges


# Draw WS cell for the specified atom
if show_atom_ws:
    center_atom = None
    elem_query = str(atom_ws_elem).strip().lower()
    if elem_query:
        candidates = [a for a in all_atoms if str(a["elem"]).strip().lower() == elem_query]
    else:
        candidates = list(all_atoms)

    if candidates:
        idx = max(0, min(int(atom_ws_index), len(candidates) - 1))
        center_atom = candidates[idx]

    if center_atom is not None:
        atom_ws_verts, atom_ws_edges = atom_ws_vertices_wireframe(
            center_atom, all_atoms, lattice_M, neighbor_range=atom_ws_neighbor_range, tol=1e-7
        )
        if len(atom_ws_verts) > 0 and len(atom_ws_edges) > 0:
            # Mark the selected center atom
            fig.add_trace(
                go.Scatter3d(
                    x=[center_atom["pos"][0]],
                    y=[center_atom["pos"][1]],
                    z=[center_atom["pos"][2]],
                    mode="markers",
                    name=f"WS 中心: {center_atom['elem']}",
                    marker=dict(size=10, color=atom_ws_color, line=dict(width=2, color="black")),
                    showlegend=True,
                )
            )
            # Draw WS edges for the selected atom
            for i, j in atom_ws_edges:
                fig.add_trace(
                    go.Scatter3d(
                        x=[atom_ws_verts[i, 0], atom_ws_verts[j, 0]],
                        y=[atom_ws_verts[i, 1], atom_ws_verts[j, 1]],
                        z=[atom_ws_verts[i, 2], atom_ws_verts[j, 2]],
                        mode="lines",
                        line=dict(color=atom_ws_color, width=atom_ws_width),
                        showlegend=False,
                        hoverinfo="none",
                    )
                )
    else:
        st.info("未找到匹配元素，无法绘制指定原子 WS 元胞。可留空元素字段并直接用索引。")


# Draw boundary frame (HCP uses a hex-prism frame; others use a parallelepiped)
if lattice_choice == HCP_STR:
    draw_hcp_hex_prism(lattice_M, fig, color=box_color, width=box_width)
else:
    draw_box(lattice_M, fig, color=box_color, width=box_width)

if show_axes:
    draw_crystal_axes(
        fig,
        lattice_M,
        axis_colors={"a": axis_color_a, "b": axis_color_b, "c": axis_color_c},
        axis_width=axis_width,
    )

if param_display_mode == "图例":
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=(
            f"a={in_a:.3f} Å, b={in_b:.3f} Å, c={in_c:.3f} Å"
            "<br>"
            f"α={in_al:.2f}°, β={in_be:.2f}°, γ={in_ga:.2f}°"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#666",
        borderwidth=1,
        font=dict(size=12, color="#111"),
    )
elif param_display_mode == "图内标注":
    draw_lattice_labels_on_cell(fig, lattice_M, in_a, in_b, in_c, in_al, in_be, in_ga)

# ================= 4. Layout =================

fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis=dict(title="X (Å)"),
        yaxis=dict(title="Y (Å)"),
        zaxis=dict(title="Z (Å)"),
    ),
    height=750,
    margin=dict(l=0, r=0, b=80, t=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="center",
        x=0.5,
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
