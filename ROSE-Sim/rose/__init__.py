
"""ROSE-Sim: minimal prototype of the ROSE neurocomputational architecture"""

from .oscillators import WilsonCowanOscillator, Sinusoid
from .workspace import SyntaxWorkspace
from . import analysis
__all__ = ['WilsonCowanOscillator', 'Sinusoid', 'SyntaxWorkspace', 'analysis']
