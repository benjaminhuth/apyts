from . import units as u

eBoundLoc = 0
eBoundPhi = 1
eBoundQoP = 2

def unit_info(idx):
    info = {
        eBoundLoc : (u.mm, "mm"),
        eBoundPhi : (u.degree, "degree"),
        eBoundQoP : (1./u.GeV, "1/GeV")
    }
    return info[idx]

kElectronMass = 0.511 * u.MeV
kSiRadiationLength =  9.370 * u.cm


