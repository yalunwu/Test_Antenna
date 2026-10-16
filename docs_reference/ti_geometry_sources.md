# TI antenna CAD provenance

The two corrected 2.4 GHz radiators are extracted from TI's original standalone antenna Gerbers. The raw DXFs are preserved alongside them for inspection. The simulator loads the derived JSON; it does not require Shapely at runtime.

| Antenna | TI resource | Archived files | Derived geometry |
| --- | --- | --- | --- |
| AN043 / CC2511 | [SWRC062, version 01.00.00.0C](https://www.ti.com/tool/download/SWRC062) | `cc2511/IFA_USB.spl`, `IFA_USB.dxf`, manufacturing README and reference archive | `cc2511/antenna_outline.json` |
| SWRU120D / CC2430 | [SWRR034 reference archive](https://www.ti.com/lit/zip/swrr034) (downloaded as SWRR034C) | `cc2430/Inverted_F_Antenna.spl`, `Inverted_F_Antenna.dxf`, manufacturing README and reference archive | `cc2430/antenna_outline.json` |

Retrieved October 13, 2026. Original artwork belongs to Texas Instruments; retain its accompanying notices and terms. The archived ZIPs preserve the source packages, not executable dependencies.

`tools/extract_ti_geometry.py` reads the inch / absolute 2.4-format Gerbers, merges filled copper regions and circular-aperture strokes, and records the original file SHA-256 in each JSON. Aperture arcs use 32 segments per circle. Derived coordinates are millimeters. The CAD copper includes outline strokes, so its bounding dimensions differ slightly from rounded application-note tables; the extractor preserves the CAD dimensions rather than tuning them to a performance result.

The CC2511 README specifies L1-L2 / L2-L3 / L3-L4 spacings of 0.25 / 0.50 / 0.25 mm and FR-4 epsilon_r = 4.5. The corrected antenna model places its RF ground 0.25 mm beneath the top copper. The CC2430 manufacturing README specifies a 0.321 mm L1-L2 spacing; the IFA model uses this RF ground depth with the application note's nominal 1 mm board-thickness baseline. Both retain simplified rectangular surrounding boards, not all circuitry and plane cutouts of the original populated boards.

Ground connections are modeled with solid PEC via barrels. AN043 uses a 10-mil via diameter from its drill table. The cropped IFA board uses 0.30 mm ground stitches as an explicit environment approximation. For the TI 77 GHz model, published 5.9-mil drill and 13.77-mil pad diameters are used; its cropped input lead length and ground width remain modeling parameters.

Regenerate and verify:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools\extract_ti_geometry.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\validate_geometry.py --native
```

The MDPI array now has a circular 0.45 mm radius pad and a point lumped feed at its center. Its 50 mm, 70 degree plates are connected thin PEC sheet facets on the Yee grid. This replaces the previous 0.8 mm solid-metal approximation; the sheet positions converge toward the specified slope as the grid is refined. The paper's copper-tape thickness is 36 micrometers, but finite copper loss, the SMA connector and the acrylic laboratory fixture are not included in this ideal antenna baseline. The fixture drawing does not specify enough wall and hole dimensions for an exact reconstruction.
