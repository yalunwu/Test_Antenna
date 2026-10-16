"""Exercise serialized geometry and reject silent, unexcited simulations."""
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import uuid
import xml.etree.ElementTree as ET

import numpy as np

import antenna_model
from openEMS import openEMS
from simulate import post_process_directory, run_openems_solver, validate_incident_wave
from solver_xml import write_solver_xml

ROOT = Path(__file__).resolve().parents[1]


class SolverExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'SimResult' / 'geometry_test_temp' / ('solver_' + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def tearDown(self):
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    def test_export_preserves_grid_and_thin_box_coordinates_for_all_models(self):
        for name in antenna_model.SUPPORTED_ANTENNAS:
            with self.subTest(model=name):
                model = antenna_model.get_antenna_model(name)
                fdtd, csx, ports, nf = model.create_simulation()
                xml = self.directory / 'model.xml'
                write_solver_xml(fdtd, csx, xml)
                settings = ET.parse(xml).find('FDTD')
                self.assertEqual(float(settings.get('endCriteria')), 1e-6)
                self.assertEqual(int(settings.get('NumberOfTimesteps')), 500000)
                loaded = openEMS()
                self.assertTrue(loaded.ReadFromXML(str(xml)))
                restored = loaded.GetCSX()
                for axis in 'xyz':
                    np.testing.assert_array_equal(restored.GetGrid().GetLines(axis), csx.GetGrid().GetLines(axis))
                originals = csx.GetAllPrimitives()
                for original, primitive in zip(originals, restored.GetAllPrimitives()):
                    if primitive.GetTypeName() not in ('Box', 'Cylinder'):
                        continue
                    np.testing.assert_array_equal(primitive.GetStart(), original.GetStart())
                    np.testing.assert_array_equal(primitive.GetStop(), original.GetStop())
                    if primitive.GetTypeName() != 'Box' or primitive.GetProperty().GetTypeString() not in ('Metal', 'Excitation'):
                        continue
                    start, stop = primitive.GetStart(), primitive.GetStop()
                    for i, axis in enumerate('xyz'):
                        if start[i] == stop[i]:
                            self.assertIn(start[i], restored.GetGrid().GetLines(axis), (name, axis, start))

    def test_zero_recordings_cannot_generate_rf_reports(self):
        for name in ('port_ut_1', 'port_it_1'):
            (self.directory / name).write_text('% t/s value\n0 0\n1e-12 0\n2e-12 0\n')
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, 'zero voltage and current'):
            post_process_directory(self.directory, antenna_model.TI24GHzAN043Antenna())
        from compare_results import read_port
        with self.assertRaisesRegex(ValueError, 'No valid incident wave'):
            read_port(self.directory, [2.44])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})

    def test_unused_saved_metal_or_source_cannot_be_post_processed(self):
        model = antenna_model.TI24GHzAN043Antenna()
        fdtd, csx, ports, nf = model.create_simulation()
        write_solver_xml(fdtd, csx, self.directory / 'openEMS_model.xml')
        for name in ('port_excite_1', 'Copper'):
            with self.subTest(property=name):
                (self.directory / 'openEMS.log').write_text(
                    f'Warning: Unused primitive (type: Box) detected in property: {name}!\n'
                )
                with self.assertRaisesRegex(ValueError, 'unused source/metal'):
                    post_process_directory(self.directory, model)
                self.assertFalse((self.directory / 'simulation_summary.json').exists())

    def test_solver_stops_at_unused_source_instead_of_running_zero_fields(self):
        model = antenna_model.TI24GHzAN043Antenna()
        fdtd, csx, ports, nf = model.create_simulation()
        xml = self.directory / 'openEMS_model.xml'
        write_solver_xml(fdtd, csx, xml)
        process = Mock(stdout=io.StringIO(
            'Warning: Unused primitive (type: Box) detected in property: port_excite_1!\n'
        ))
        with patch('simulate.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'absent from the solver grid'):
                run_openems_solver(self.directory, xml)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)

    def test_invalid_incident_wave_is_rejected(self):
        for wave in ([0, 0], [1, np.nan], [1, np.inf]):
            with self.subTest(wave=wave):
                with self.assertRaisesRegex(ValueError, 'no valid incident wave'):
                    validate_incident_wave(Mock(number=1, uf_inc=np.array(wave)))


if __name__ == '__main__':
    unittest.main()
