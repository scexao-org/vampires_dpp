from astropy.io import fits
from pathlib import Path

from vampires_dpp.headers import dict_from_header, observation_table

TEST_DIR = Path(__file__).parent
TEST_FILE = Path(TEST_DIR, "data", "VMPA00021059.fits")


def test_dict_from_header():
    summary = dict_from_header(TEST_FILE)
    assert summary["SIMPLE"]
    assert summary["BITPIX"] == 16
    assert summary["NAXIS"] == 3
    assert summary["NAXIS1"] == 256
    assert summary["NAXIS2"] == 256
    assert summary["NAXIS3"] == 139
    assert summary["EXTEND"]
    assert summary["BSCALE"] == 1
    assert summary["BZERO"] == 32768
    assert summary["DATE-OBS"] == "2022-02-24"
    assert summary["OBJECT"] == "ABAUR"
    assert summary["UT"] == "07:22:36.124"
    assert summary["UT-END"] == "07:22:34:880"
    assert summary["UT-STR"] == "07:21:24:254"
    assert summary["HST"] == "21:22:36.124"
    assert summary["HST-END"] == "21:22:34:887"
    assert summary["HST-STR"] == "21:21:24:262"
    assert summary["MJD"] == 59634.30736254
    assert summary["MJD-END"] == 59634.307338
    assert summary["MJD-STR"] == 59634.306528
    assert summary["FRAMEID"] == "VMPA00021059"
    assert summary["EXPTIME"] == 0.25
    assert summary["DATA-TYP"] == "ACQUISITION"
    assert summary["OBSERVAT"] == "NAOJ"
    assert summary["TELESCOP"] == "Subaru"
    assert summary["INSTRUME"] == "VAMPIRES"
    assert summary["TIMESYS"] == "UTC"
    assert summary["PROP-ID"] == "o22193"
    assert (
        summary["OBSERVER"]
        == '"[Hilo] Bottom, Guyon, Lozi, Deo, Lucas [Summit] Barjot"'
    )
    assert summary["OBS-ALOC"] == "Observation"
    assert summary["OBS-MOD"] == "IMAG_POL"
    assert summary["RADESYS"] == "FK5"
    assert summary["LONPOLE"] == 180.0
    assert summary["RA"] == "04:55:45.674"
    assert summary["DEC"] == "+30:33:07.39"
    assert summary["EQUINOX"] == 2000.0
    assert summary["TELFOCUS"] == "NS_IR"
    assert summary["FOC-VAL"] == -0.12
    assert summary["AIRMASS"] == 1.196
    assert summary["ZD"] == 33.31075
    assert summary["AZIMUTH"] == 295.86528
    assert summary["ALTITUDE"] == 56.68925
    assert summary["FOC-POS"] == "Nasmyth-IR"
    assert summary["AUTOGUID"] == "OFF"
    assert summary["M2-TYPE"] == "IR"
    assert summary["OUT-HUM"] == 3.4
    assert summary["OUT-PRS"] == 623.5
    assert summary["OUT-TMP"] == 276.25
    assert summary["OUT-WND"] == 9.1
    assert summary["U_CAMERA"] == 1.0
    assert summary["U_AQTINT"] == 250000.0
    assert summary["U_AQNCYC"] == 140.0
    assert summary["U_FLCOFF"] == 0.0
    assert summary["U_AQSDEL"] == 100000.0
    assert summary["U_AQDTIM"] == 250000.0
    assert summary["U_HWPANG"] == 45.0
    assert summary["U_NLOOPS"] == 999.0
    assert summary["U_LOOPIT"] == 14.0
    assert summary["U_NPOLST"] == 4.0
    assert summary["U_PLSTIT"] == 2.0
    assert summary["U_ORGDIR"] == "/mnt/data/20220224/"
    assert summary["U_OGFNAM"] == "ABAur_03_20220224_750-50_LyotStop_0"
    assert summary["U_OGFNUM"] == "53"
    assert summary["U_EMGAIN"] == 300.0
    assert summary["U_QWP1"] == 128.0
    assert summary["U_QWP2"] == 168.0
    assert summary["U_FILTER"] == "750-50"
    assert summary["U_MASK"] == "LyotStop"
    assert summary["U_MANGLE"] == 1.0
    assert summary["U_AOHWP"]
    assert summary["D_SADC"] == "IN"
    assert summary["D_SADCA1"] == 161.55798
    assert summary["D_SADCA2"] == 157.55796
    assert summary["D_SADCDC"] == "+30:33:04"
    assert summary["D_SADCFC"] == 1.0
    assert summary["D_SADCMD"] == "ADI"
    assert summary["D_SADCPA"] == -39.0
    assert summary["D_SADCP"] == 163.025
    assert summary["D_SADCRA"] == "04:55:45.9"
    assert summary["D_SADCST"] == "SYNC"
    assert summary["D_BS1"] == "SCEXAO"
    assert summary["D_BS1P"] == 348.72
    assert summary["D_BS1S"] == 35708928.0
    assert summary["D_BS2"] == "MIRROR"
    assert summary["D_BS2P"] == 152.63
    assert summary["D_BS2S"] == 15629312.0
    assert summary["D_ENSHUT"] == "OPEN"
    assert summary["D_BNCHI"] == 2.0
    assert summary["D_BNCHO"] == 100.0
    assert summary["D_BNCTI"] == 12.5
    assert summary["D_BNCTO"] == 0.0
    assert summary["D_IMRANG"] == 92.85608
    assert summary["D_IMRDEC"] == "+30:33:04.29"
    assert summary["D_IMRMOD"] == "ADI"
    assert summary["D_IMRPAD"] == -118.507
    assert summary["D_IMRPAP"] == -39.0
    assert summary["D_IMRRA"] == "04:55:45.850"
    assert summary["D_IMR"] == "TRACK"
    assert summary["D_ADFG"] == 0.0
    assert summary["D_DMCMTX"] == "ao188cmtx.oct"
    assert summary["D_DMGAIN"] == 10.0
    assert summary["D_HDFG"] == 1.0
    assert summary["D_HTTG"] == 1.0
    assert summary["D_LDFG"] == 0.0
    assert summary["D_LOOP"] == "ON"
    assert summary["D_LTTG"] == 0.0
    assert summary["D_PSUBG"] == 0.01
    assert summary["D_STTG"] == 0.0
    assert summary["D_TTCMTX"] == "ao188ttctrl.oct"
    assert summary["D_TTGAIN"] == 0.001
    assert summary["D_WTTG"] == 0.0
    assert summary["DATE"] == "2022-02-24T07:22:36"
    assert summary["U_FLCSTT"] == 1
    assert (
        summary["COMMENT"]
        == "FITS (Flexible Image Transport System) format is defined in 'Astronomy, and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
    )
