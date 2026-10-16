"""Shared solver/preview geometry for the TI Gerber-derived PCB radiators."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import openems_config
from CSXCAD import ContinuousStructure
from openEMS import openEMS

ROOT = Path(__file__).resolve().parent


def load_outline(directory):
    return json.loads((ROOT / "docs_reference" / directory / "antenna_outline.json").read_text(encoding="utf-8"))


def circle_points(x, y, radius, count=64):
    angle = np.arange(count) * (2 * np.pi / count)
    return np.c_[x + radius * np.cos(angle), y + radius * np.sin(angle)]


def rectangle(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


class GerberPCBGeometry:
    """One geometry definition used for FDTD and all three previews.

    The radiator comes from TI's complete standalone Gerber, including its
    aperture strokes. The surrounding rectangular board remains a controlled
    approximation, rather than a reproduction of the populated reference PCB.
    """
    def _initialize_outline(self, directory, align_left=None):
        self.reference_outline = load_outline(directory)
        self.reference_geometry_directory = directory
        lo, _, hi, _ = self.reference_outline["bounds_mm"]
        shift = align_left - lo if align_left is not None else -(lo + hi) / 2
        self.radiator_polygons = [np.array(p) + [shift, 0] for p in self.reference_outline["polygons_mm"]]
        self.feed_center = np.array(self.reference_outline["feed_center_mm"]) + [shift, 0]
        self.ground_z = self.sub_h - self.rf_ground_depth
        if not 0 < self.rf_ground_depth <= self.sub_h:
            raise ValueError("RF ground depth must be positive and no greater than PCB thickness")
        self.board_x_min, self.board_x_max = -self.sub_w / 2, self.sub_w / 2
        bounds = np.concatenate(self.radiator_polygons)
        if bounds[:, 0].min() < self.board_x_min or bounds[:, 0].max() > self.board_x_max:
            raise ValueError("Radiator extends beyond the substrate")
        self.sub_y_max = float(bounds[:, 1].max()) + self.radiator_edge_margin
        self.sub_l = self.sub_y_max - self.sub_y_min

    def _get_radiator_polygons(self):
        return self.radiator_polygons

    def _get_ground_planes(self):
        planes = [(self.ground_z, rectangle(self.board_x_min, self.board_x_max, self.sub_y_min, 0))]
        if self.coplanar_ground:
            x = self.feed_center[0]
            clearance = self.W2 / 2 + self.L8
            planes += [(self.sub_h, rectangle(self.board_x_min, x - clearance, self.sub_y_min, 0)),
                       (self.sub_h, rectangle(x + clearance, self.board_x_max, self.sub_y_min, 0))]
        return planes

    def create_simulation(self, end_criteria=1e-6, nr_ts=500000, active_channels=1):
        fdtd = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        fdtd.SetGaussExcite(self.f_center, 1e9)
        fdtd.SetBoundaryCond(["PML_8"] * 6)
        csx = ContinuousStructure()
        fdtd.SetCSX(csx)
        mesh = csx.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        substrate = csx.AddMaterial("FR4", epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        copper = csx.AddMetal("Copper")
        substrate.AddBox([self.board_x_min, self.sub_y_min, 0],
                         [self.board_x_max, self.sub_y_max, self.sub_h], priority=0)
        for z, polygon in self._get_ground_planes():
            copper.AddLinPoly(polygon.T, "z", z, 0.0, priority=10)
        for polygon in self.radiator_polygons:
            copper.AddLinPoly(polygon.T, "z", self.sub_h, 0.0, priority=10)
        for x, y, radius in self.ground_vias:
            copper.AddCylinder([x, y, self.ground_z], [x, y, self.sub_h], radius, priority=11)
        x, y = self.feed_center
        # A narrow vertical lumped port at the actual Gerber feed contact.
        # Its lower terminal contacts the continuous L2 reference plane.
        port = fdtd.AddLumpedPort(1, 50, [x, y, self.ground_z], [x, y, self.sub_h],
                                 "z", 1, priority=20, edges2grid="xy")
        res = self.lambda0_mm / self.cells_per_wavelength
        pad = self.lambda0_mm * .45
        points = np.concatenate(self.radiator_polygons)
        # Resolve the source's smallest 0.46/0.50 mm traces locally. Do not
        # seed every sampled arc vertex: that would create near-zero cells.
        local_res = min(.15, res)
        for axis, column, limits in (("x", 0, [self.board_x_min, self.board_x_max]),
                                     ("y", 1, [self.sub_y_min, self.sub_y_max])):
            low, high = points[:, column].min() - .5, points[:, column].max() + .5
            local = np.linspace(low, high, int(np.ceil((high-low)/local_res))+1)
            # Include the port and ground boundary exactly; keep all features
            # separated enough for a practical timestep.
            mandatory = np.array(limits + [self.feed_center[column], 0.0])
            local = local[np.min(abs(local[:, None]-mandatory[None, :]), axis=1) > .02]
            raw = np.r_[limits[0]-pad, local, mandatory, limits[1]+pad]
            mesh.AddLine(axis, self._smooth_mesh(raw, res, ratio=1.3))
        z = np.r_[-pad, 0.0, np.linspace(self.ground_z, self.sub_h, 5), self.sub_h+pad]
        mesh.AddLine("z", self._smooth_mesh(z, res, ratio=1.3))
        return fdtd, csx, [port], fdtd.CreateNF2FFBox()

    def draw_layout(self, output_path):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        ax = axes[0]
        ax.add_patch(PlotPolygon(rectangle(self.board_x_min, self.board_x_max, self.sub_y_min, self.sub_y_max),
                                 facecolor="#edf2f7", edgecolor="#64748b"))
        for z, polygon in self._get_ground_planes():
            ax.add_patch(PlotPolygon(polygon, facecolor="#a8b5c3", alpha=.45))
        for polygon in self.radiator_polygons:
            ax.add_patch(PlotPolygon(polygon, facecolor="#d97706", edgecolor="#78350f", lw=.7))
        for x, y, radius in self.ground_vias:
            ax.add_patch(plt.Circle((x, y), radius, facecolor="#334155"))
        ax.plot(*self.feed_center, "ro", ms=4, label="50 ohm feed")
        ax.set(xlim=(self.board_x_min-1, self.board_x_max+1),
               ylim=(self.sub_y_min-1, self.sub_y_max+1), xlabel="X (mm)", ylabel="Y (mm)",
               title="Original TI Gerber copper")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc='lower left')
        side = axes[1]
        side.fill_between([self.sub_y_min, self.sub_y_max], 0, self.sub_h, color="#dbeafe")
        side.plot([self.sub_y_min, 0], [self.ground_z]*2, color="#78350f", lw=3, label="L2 RF ground")
        side.plot([-.5, self.sub_y_max-self.radiator_edge_margin], [self.sub_h]*2,
                  color="#d97706", lw=3, label="Top radiator")
        side.plot([self.feed_center[1]]*2, [self.ground_z, self.sub_h], "r-", lw=2, label="Port")
        side.set(xlabel="Y (mm)", ylabel="Z (mm)", title=f"RF ground depth: {self.rf_ground_depth:.3f} mm",
                 ylim=(-.1, self.sub_h+.3))
        side.legend(fontsize=8)
        fig.suptitle(self.title, fontsize=12)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_3d_view(self, output_path):
        fig = plt.figure(figsize=(10, 7), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        board = rectangle(self.board_x_min, self.board_x_max, self.sub_y_min, self.sub_y_max)
        for z, polygon, color, alpha in ([(0, board, "#94a3b8", .2), (self.sub_h, board, "#94a3b8", .2)] +
                                       [(z, p, "#78350f", .6) for z, p in self._get_ground_planes()] +
                                       [(self.sub_h, p, "#d97706", 1) for p in self.radiator_polygons]):
            ax.add_collection3d(Poly3DCollection([np.c_[polygon, np.full(len(polygon), z)]],
                                               facecolor=color, alpha=alpha))
        for x, y, radius in self.ground_vias:
            ax.plot([x, x], [y, y], [self.ground_z, self.sub_h], color="#475569", lw=3)
        x, y = self.feed_center
        ax.plot([x, x], [y, y], [self.ground_z, self.sub_h], "r-", lw=3)
        ax.set(xlim=(self.board_x_min, self.board_x_max), ylim=(self.sub_y_min, self.sub_y_max),
               zlim=(0, self.sub_h+2), xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=self.title)
        ax.view_init(30, -55)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_interactive_3d(self, output_path):
        polygons = [{"z": z, "points": p.tolist(), "color": "#78350f"} for z, p in self._get_ground_planes()]
        polygons += [{"z": self.sub_h, "points": p.tolist(), "color": "#d97706"} for p in self.radiator_polygons]
        data = json.dumps({"polygons": polygons, "vias": self.ground_vias, "feed": self.feed_center.tolist(),
                           "groundZ": self.ground_z, "topZ": self.sub_h})
        html = '''<!doctype html><html><head><meta charset="utf-8"><title>TI reference geometry</title>
<style>body{margin:0;background:#0f172a;color:#e2e8f0;font-family:system-ui}#label{position:absolute;top:12px;left:16px;max-width:440px}</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script></head><body>
<div id="label">TITLE<br>Original TI Gerber outline; L2 ground and ground vias.<br>Drag to rotate; scroll to zoom.</div><script>
const data=DATA,scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.1,1000);
camera.position.set(35,-60,45); const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera,renderer.domElement);controls.target.set(0,-10,0);controls.update();
scene.add(new THREE.AmbientLight(0xffffff,.9));
data.polygons.forEach(p=>{const s=new THREE.Shape();p.points.forEach((v,i)=>i?s.lineTo(...v):s.moveTo(...v));s.closePath();const m=new THREE.Mesh(new THREE.ShapeGeometry(s),new THREE.MeshBasicMaterial({color:p.color,side:THREE.DoubleSide}));m.position.z=p.z;scene.add(m);});
data.vias.forEach(v=>{const g=new THREE.CylinderGeometry(v[2],v[2],data.topZ-data.groundZ,24);g.rotateX(Math.PI/2);const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0x94a3b8}));m.position.set(v[0],v[1],(data.topZ+data.groundZ)/2);scene.add(m);});
const port=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...data.feed,data.groundZ),new THREE.Vector3(...data.feed,data.topZ)]);scene.add(new THREE.Line(port,new THREE.LineBasicMaterial({color:0xef4444})));
function draw(){requestAnimationFrame(draw);controls.update();renderer.render(scene,camera)}draw();addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
</script></body></html>'''
        Path(output_path).write_text(html.replace("TITLE", self.title).replace("DATA", data), encoding="utf-8")
