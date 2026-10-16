"""Source fidelity, conductor contact and thin-sheet geometry regression checks."""
import hashlib
import json
import uuid
import unittest
from pathlib import Path

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import antenna_model
from tools.extract_ti_geometry import read_gerber

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_ti_outlines_match_original_gerber(self):
        for name, directory in (("ti_24ghz_an043", "cc2511"), ("ti_24ghz_swru120d", "cc2430")):
            with self.subTest(model=name):
                model = antenna_model.get_antenna_model(name)
                ref = model.reference_outline
                source = ROOT / "docs_reference" / directory / ref["source_file"]
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), ref["source_sha256"])
                origin = np.array(ref["source_origin"]) * ref["source_coordinate_unit_mm"]
                shift = model.feed_center[0]
                exact = translate(read_gerber(source), -origin[0]+shift, -origin[1])
                modeled = unary_union([Polygon(p) for p in model.radiator_polygons])
                self.assertLess(exact.hausdorff_distance(modeled), 1e-7)
                self.assertLess(exact.symmetric_difference(modeled).area, 1e-5)

    def test_radiators_and_vias_are_inside_boards(self):
        for name in ("ti_24ghz_an043", "ti_24ghz_swru120d"):
            model = antenna_model.get_antenna_model(name)
            for p in model.radiator_polygons:
                self.assertGreaterEqual(p[:, 0].min(), model.board_x_min)
                self.assertLessEqual(p[:, 0].max(), model.board_x_max)
                self.assertLessEqual(p[:, 1].max(), model.sub_y_max)
            self.assertGreater(model.ground_z, 0)
            self.assertLess(model.ground_z, model.sub_h)

    def test_ti_feed_terminals_and_ground_connections(self):
        for name in ("ti_24ghz_an043", "ti_24ghz_swru120d"):
            model = antenna_model.get_antenna_model(name)
            feed = Point(model.feed_center)
            top = unary_union([Polygon(p) for p in model.radiator_polygons])
            bottom = unary_union([Polygon(p) for z, p in model._get_ground_planes() if z == model.ground_z])
            self.assertTrue(top.covers(feed))
            self.assertTrue(bottom.covers(feed))
            ground_top = unary_union([top] + [Polygon(p) for z, p in model._get_ground_planes() if z == model.sub_h])
            for x, y, radius in model.ground_vias:
                self.assertTrue(bottom.covers(Point(x, y)))
                self.assertTrue(ground_top.covers(Point(x, y)))

    def test_ti77_transition_and_vias(self):
        model = antenna_model.TI77GHzRXAntenna()
        self.assertAlmostEqual(model.y_trans_end-model.y_trans_start,
                               model.A1_gcpw_len+model.A2_trans_len)
        vias = model._get_ground_vias()
        self.assertEqual(len(vias), 8 * model.num_channels)
        grounds = unary_union([Polygon(p) for xc in model.channel_offsets_x
                               for p in model._get_gcpw_ground_polygons(xc)])
        for x, y in vias:
            self.assertTrue(grounds.covers(Point(x, y).buffer(model.via_pad_diameter/2)))
        fdtd, csx, ports, nf = model.create_simulation()
        self.assertFalse(csx.Update())
        copper = csx.GetPropertyByCoordPriority([vias[0][0], vias[0][1], model.sub_h/2])
        self.assertEqual(copper.GetName(), "Copper")
        self.assertAlmostEqual(ports[0].start[1], model.y_gcpw_start)

    def test_mdpi_matching_disk_has_no_square_corners(self):
        model = antenna_model.MDPI24GHzPatchArrayAntenna()
        disk = Polygon(model._get_matching_disk())
        self.assertLess(abs(disk.area/(np.pi*model.r1**2)-1), .002)
        fdtd, csx, ports, nf = model.create_simulation()
        self.assertFalse(csx.Update())
        prop = csx.GetPropertyByCoordPriority([.9*model.r1, model.y_port+.9*model.r1, model.sub_h])
        self.assertTrue(prop is None or prop.GetName() != "copper")
        self.assertAlmostEqual(ports[0].start[0], ports[0].stop[0])
        self.assertAlmostEqual(ports[0].start[1], ports[0].stop[1])
        self.assertFalse(any("disk" in b[0] for b in model._get_trace_boxes()))

    def test_mdpi_thin_plates_are_connected_and_follow_source_flare(self):
        model = antenna_model.MDPI24GHzPatchArrayAntenna(with_plates=True)
        fdtd, csx, ports, nf = model.create_simulation()
        self.assertFalse(csx.Update())
        grid = csx.GetGrid()
        facets = model._get_plate_facets(grid.GetLines("y"), grid.GetLines("z"))
        previous = (model.sub_l/2, 0)
        yres = np.diff(grid.GetLines("y")).max()
        zres = np.diff(grid.GetLines("z")).max()
        for yl, yr, zl, zr in facets:
            np.testing.assert_allclose([yl, zl], previous, atol=1e-9)
            self.assertTrue(yl == yr or zl == zr)
            ideal = model.sub_l/2+(zl+zr)/2/np.tan(model.theta_rad)
            self.assertLessEqual(abs((yl+yr)/2-ideal), yres+zres/np.tan(model.theta_rad))
            previous = (yr, zr)
        np.testing.assert_allclose(previous, [model.sub_l/2+model.dy_plate, model.dz_plate])
        # A middle face must be PEC in the actual CSX model, not an absent thin volume.
        yl, yr, zl, zr = facets[len(facets)//2]
        prop = csx.GetPropertyByCoordPriority([0, (yl+yr)/2, (zl+zr)/2])
        self.assertEqual(prop.GetName(), "copper")
        # Both sides must be on bit-identical native grid planes. Approximate
        # symmetry is insufficient for zero-thickness PEC faces.
        for primitive in csx.GetAllPrimitives():
            if primitive.GetPriority() != 15:
                continue
            start, stop = primitive.GetStart(), primitive.GetStop()
            for axis, index in (("y", 1), ("z", 2)):
                if start[index] == stop[index]:
                    self.assertTrue(np.any(grid.GetLines(axis) == start[index]))

    def test_all_models_export_valid_meshes(self):
        for name in antenna_model.SUPPORTED_ANTENNAS:
            model = antenna_model.get_antenna_model(name)
            fdtd, csx, ports, nf = model.create_simulation()
            for axis in "xyz":
                delta = np.diff(csx.GetGrid().GetLines(axis))
                self.assertTrue(np.all(delta > 1e-8), name)

    def test_old_runs_cannot_be_relabelled_with_corrected_geometry(self):
        from simulate import post_process_directory
        scratch = ROOT / 'SimResult' / 'geometry_test_temp'
        scratch.mkdir(parents=True, exist_ok=True)
        directory = scratch / ('baseline_guard_' + uuid.uuid4().hex)
        directory.mkdir()
        try:
            self.assertTrue(directory.resolve().is_relative_to(scratch.resolve()))
            original = json.dumps({'antenna_model': {'name': 'ti_24ghz_an043'}})
            summary = directory / 'simulation_summary.json'
            summary.write_text(original)
            with self.assertRaisesRegex(ValueError, 'Run a new simulation'):
                post_process_directory(directory, antenna_model.TI24GHzAN043Antenna())
            self.assertEqual(summary.read_text(), original)
            self.assertEqual(len(list(directory.iterdir())), 1)
        finally:
            summary.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
