# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 07:07:49 2021

@author: jadsadek
"""

from collections import UserList
from itertools import combinations
from logging import currentframe, raiseExceptions
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.lines as lin 
import matplotlib.colors as col
from slog.geometry_module import Node, Segment, Vector, Line, Polygon, NUMBERS, SCALE, INFTY, _axis_setup
from PyQt5 import QtGui
import ctypes
import os 
import platform


# Constants

# Geotechnical classes

class Geometry(Polygon):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.update_display_properties(linewidth= 2*SCALE, fill=False)


class Facies(Polygon):
    def __init__(self, nodes, ConstitutiveLaw = None):
        super().__init__(nodes)
        self.ConstitutiveLaw = ConstitutiveLaw
        self.update_display_properties(linestyle='--')
    
    def intersect_geometry(self, geom):
        intersection = []
        for polygon in self.intersect(geom):
            facies = Facies(polygon, self.ConstitutiveLaw)
            facies.update_display_properties(**self._display_properties)
            facies.update_display_properties(fill=True)
            intersection.append(facies)
        return intersection
    
            
class Geology(UserList):

    def __init__(self, initlist):
        super().__init__(initlist)
        if not self.is_disjoint():
            raise ValueError("Facies are not disjoint")

    def is_disjoint(self):
        for fac1, fac2 in combinations(self, 2):
            if len(fac1.intersect(fac2))!=0:
                print(fac1, fac2, fac1.intersect(fac2))
                return False
        return True

    def _distribute_colors(self):
        nFacies = self.__len__()
        for (i, facies) in enumerate(self):
            facies.update_display_properties(facecolor=(*col.hsv_to_rgb((0.05, 0.5 + i/2/nFacies, 0.5)),0.85))

    def draw(self, ax=None,  **kwargs):
        ax, current_figure = _axis_setup(ax)
        for facies in self:
            facies.draw(ax, **kwargs)

    def draw_legend(self, ax=None,  highlight=None, **kwargs):
        ax, current_figure = _axis_setup(ax)
        x, y = ax.transData.transform((0,1))
        current_y = y - 20
        for facies in self:
            xp0, yp0 = ax.transData.inverted().transform((x + 5, current_y-15))
            xp1, yp1 = ax.transData.inverted().transform((x + 25, current_y + 5))
            xp, yp = ax.transData.inverted().transform((x + 30, current_y))
            ax.text(xp, yp, facies.ConstitutiveLaw.__repr__(), fontsize=12, va='top')
            if facies == highlight:
                current_y = current_y - 20
                xp, yp = ax.transData.inverted().transform((x+10,current_y))
                ax.text(xp, yp, facies.ConstitutiveLaw, fontsize=10, va='top')
                current_y = current_y - 70
            else:
                current_y = current_y - 30
            ax.add_patch(patch.Polygon([[xp0, yp0], [xp1, yp0], [xp1, yp1], [xp0, yp1]], hatch=facies._patch.get_hatch(), linewidth=facies._patch.get_linewidth(), facecolor=facies._patch.get_facecolor()))
    def create_terrain_elements(self, geom):
        intersection =[]
        for facies in self:
            intersection.extend(facies.intersect_geometry(geom))
        return GeologyModel(intersection)


class WaterZone(Polygon):
    def __init__(self, nodes, CLvitesse = None, CLpression = None):
        self._nodes = nodes
        self.CLvitesse = CLvitesse
        self.CLpression = CLpression
        self.update_display_properties(fill=True, facecolor='b', alpha=0.3)


class Terrain:
    def __init__(self, geom, geol, water=None):
        self.geometry = geom
        self.geology = geol
        self.water = water
        self.geology_model = None

    def compute_geology_model(self):
        self.geology_model = self.geology.create_terrain_elements(self.geometry)

    def draw(self, ax=None, **kwargs):
        ax, current_figure = _axis_setup(ax)
        self.geology_model.draw(ax,**kwargs)
        self.geology.draw(ax, **kwargs)
        self.geometry.draw(ax, **kwargs)
        

class Structure1d(Line):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.update_display_properties (linewidth=2*SCALE)
    

class Structure2d(Polygon):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.update_display_properties (linewidth=2*SCALE)
    

class Clou(Structure1d):
    def __init__(self, A, B):
        super().__init__([A, B])
        self.update_display_properties(linewidth = 3*SCALE)

class Bache(Structure1d):
    def __init__(self, nodeList):
        super().__init__(nodeList)
        self.update_display_properties(linewidth = 2.5*SCALE)

class Wall(Structure2d):
    def __init__(self, nodeList):
        super().__init__(nodeList)
        self.update_display_properties(facecolor = 'k')


class GeologyModel(Geology):
    def __init__(self, fac):
        super().__init__(fac)

    def create_terrain_elements(self, geom):
        raise TypeError(f"{type(self).__name__} is not meant to create terrain elements. Please use Geology instead.")

    def calculate_interfaces(self, structure1dList):
        interfaces = []
        for struct in structure1dList:
            if isinstance(struct, Structure1d):
                for fac in self:
                    interfaces.extend(list((s, fac, struct) for s in struct.intersect_polygon(fac)))
            elif isinstance(struct, Structure2d):
                for fac in self:
                    interfaces.extend(list((s, fac, struct) for s in struct.boundary.intersect_polygon(fac)))
        return interfaces
    

class Configuration:
    def __init__(self, terr, structList):
        self.terrain = terr
        self.structures = list(structList)

    def draw(self, ax=None, **kwargs):
        ax, current_figure = _axis_setup(ax)
        self.terrain.draw(ax, **kwargs)
        for struct in self.structures:
            struct.draw(ax, **kwargs)

    def show(self, xmin=0, ymin=0, xmax=1, ymax=1, ax=None, **kwargs):
        ax, current_figure = _axis_setup(ax)
        self.draw(ax=ax, **kwargs)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.show()


class ConstitutiveLaw:
    def __init__(self, mat, rho, E):
        self.material = mat
        self.young_modulus = E
        self.density = rho
    def __repr__(self):
        return str(self.material)
    def __str__(self):
        return f"Material: {self.material}\nDensity: {self.density} kg/m$^3$ \nYoung's Modulus: {self.young_modulus} GPa"



class Ouvrage(UserList): 
    def show(self, xmin=0, ymin=0, xmax=1, ymax=1, ax=None, **kwargs):           
        ax, fig = _axis_setup(ax)
        fig.ouvrage = self
        fig.current_index = None
        config_name = f"Please use the Up and Down arrow key\n or scroll up and down \n to cycle through the {self.__len__()} configuration" + ("s" if self.__len__()>1 else "")
        fig.txt = fig.text(0.5, 0.5, config_name, fontsize = 12, ha='center', va='center')
        fig.highlight = None
        fig.patch.set_facecolor('lightgrey')
        def on_key_release(event):
            if fig.ouvrage.__len__() == 0: return True
            if event.key == 'up' or (isinstance(event, matplotlib.backend_bases.MouseEvent) and event.button == 'up'):
                if fig.current_index is None:
                    fig.current_index = 0
                else:
                    fig.current_index = (fig.current_index + 1) % fig.ouvrage.__len__()
            elif event.key == 'down' or (isinstance(event, matplotlib.backend_bases.MouseEvent) and event.button == 'down'):
                if fig.current_index is None:
                    fig.current_index = 0
                else:
                    fig.current_index = (fig.current_index - 1) % fig.ouvrage.__len__()
            else:
                return True
            fig.highlight = None    
            fig.clf()
            ax = fig.add_axes([0.05, 0.05, 0.64, 0.9])
            ax.set_ylim([ymin,ymax])
            ax.set_xlim([xmin,xmax])
            ax.set_aspect('equal', adjustable='datalim')
            fig.ouvrage[fig.current_index].draw(ax, **kwargs)
            fig.ax2 = fig.add_axes([0.70, 0.3, 0.29, 0.65], xticks=[], yticks=[])
            fig.ouvrage[fig.current_index].terrain.geology_model.draw_legend(fig.ax2, **kwargs)
            fig.text(0.85, 0.2, f"Configur$\\alpha$tion {fig.current_index+1}/{self.__len__()}", fontsize=12, ha='center')
            fig.text(1, 0, f"SLOG, Jad Sadek", fontsize=7, ha='right', va='bottom')
            fig.canvas.draw()
            return True

        def on_button_press(event):
            if fig.highlight is not None:
                fig.highlight._patch.set_hatch('')
                fig.highlight = None
                fig.ax2.cla()
                fig.ax2.set_xticks([])
                fig.ax2.set_yticks([])
                fig.ouvrage[fig.current_index].terrain.geology_model.draw_legend(fig.ax2, **kwargs)
            if fig.current_index is None or event.xdata is None or event.ydata is None or event.inaxes!=fig.axes[0]:
                fig.canvas.draw()
                return True
            P = Node(event.xdata, event.ydata)
            terr = fig.ouvrage[fig.current_index].terrain
            if P.in_polygon(terr.geometry):
                for fac in terr.geology_model:
                    if P.in_polygon(fac):
                        print(fac)
                        fig.highlight = fac
                        fac._patch.set_hatch('///')
                        fig.ax2.cla()
                        fig.ax2.set_xticks([])
                        fig.ax2.set_yticks([])
                        fig.ouvrage[fig.current_index].terrain.geology_model.draw_legend(fig.ax2, highlight=fac, **kwargs)
                        break
            fig.canvas.draw()

        def on_resize(event):
            if len(fig.axes) >= 2:
                fig.ax2.cla()
                fig.ax2.set_xticks([])
                fig.ax2.set_yticks([])
                fig.ouvrage[fig.current_index].terrain.geology_model.draw_legend(fig.ax2, highlight=fig.highlight, **kwargs)

        fig.canvas.mpl_connect('key_release_event', on_key_release)
        fig.canvas.mpl_connect('scroll_event', on_key_release)
        fig.canvas.mpl_connect('button_press_event', on_button_press)
        fig.canvas.mpl_connect('resize_event', on_resize)
        figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        figManager.set_window_title('SLOG')
        figManager.window.setWindowIcon(QtGui.QIcon(os.path.dirname(__file__) + '/SLOG.svg'))
        if platform.system() == 'Windows':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('slog')
        plt.show()


#french aliases
Noeud = Node
Geometrie = Geometry
Geologie = Geology


if __name__ == "__main__":
    argile = ConstitutiveLaw('argile', 3000, 1)
    calcaire = ConstitutiveLaw('calcaire', 2000, 10)
    P0 = Node(0, 0)
    P1 = Node(1, 0)
    P2 = Node(1, 0.8)
    P3 = Node(0.7, 0.8)
    P4 = Node(0.6, 0.5)
    P5 = Node(0.4, 0.5)
    P6 = Node(0.3, 0.8)
    P7 = Node(0, 0.8)
    geom = Geometry([P0, P1, P2, P3, P4, P5, P6, P7])
    geom2 = Geometry([P0, P1, P2, P3, P4])
 

    Q4 = Node(0.5, 0.5)
    Q5 = Node(-1, 1)
    Q2b = Node(2, 2)
    Q1b = Node(-1, 2)
    f1 = Facies([P0, P1, P2, Q4, P7], argile)
    f2 = Facies([P7, Q4, P2], calcaire)
    geol = Geology([f1, f2])
    cl1 = Clou(Node(0.65, 0.65), Node(0.75,0.55))
    cl2 = Clou(P3, Node(0.9, 0.6))
    terr = Terrain(geom, geol)
    terr2 = Terrain(geom2, geol)
    geol._distribute_colors()
    terr.compute_geology_model()
    terr2.compute_geology_model()
    print(terr.geology_model.calculate_interfaces([cl1, cl2]))
    config = Configuration(terr, [cl1, cl2])
    config2 = Configuration(terr2,  [cl1, cl2])
    ouv = Ouvrage((config,config2))
    ouv.show(xmin=-0.1, xmax=1.5, ymin=-0.1, ymax=1)


