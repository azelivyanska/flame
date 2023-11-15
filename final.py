import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation, ArtistAnimation

#жесткое описание вектора и его своиств(сложение там, умножение)
class Vector(list):
    def __init__(self, *el):
        for e in el:
            self.append(e)

    def __add__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] + other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self + other

    def __sub__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] - other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self - other

    def __mul__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] * other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self * other

    def __truediv__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] / other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self / other

    def __pow__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] ** other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self ** other

    def __mod__(self, other):
        return sum((self - other) ** 2) ** 0.5

    def mod(self):
        return self % Vector.emptyvec(len(self))

    def dim(self):
        return len(self)

    def __str__(self):
        if len(self) == 0:
            return "Empty"
        r = [str(i) for i in self]
        return "< " + " ".join(r) + " >"

    def _ipython_display_(self):
        print(str(self))

    @staticmethod
    def emptyvec(lens=2, n=0):
        return Vector(*[n for i in range(lens)])

    @staticmethod
    def randvec(dim,r1,r2):
        m=[r1,r2]
        return Vector(*[random.random()*m[i] for i in range(2)])
#жесткое описание точки и его свойств
class Point:
    def __init__(self, coords, mass=1.0, q=1.0 ,speed = None, **properties):
        self.coords=coords
        if speed is None:
            self.speed = Vector(*[0 for i in range(len(coords))])#не ввел скорость-она 0
        else:
            self.speed = speed
        self.acc = Vector(*[0 for i in range(len(coords))])#начальное ускорение = 0
        self.mass = mass
        self.__params__ = ["coords", "speed", "acc", "q"] + list(properties.keys())
        self.q = q
        for prop in properties:
            setattr(self, prop, properties[prop])


    def move(self, dt):
        self.coords = self.coords + self.speed * dt
        #изменение координат и там ниже еще скорости и ускорения
    def otr(self):
        self.speed[0]=-self.speed[0]


    def accelerate(self, dt):
        self.speed = self.speed + self.acc * dt


    def accinc(self, force):
        self.acc = self.acc + force / self.mass


    def clean_acc(self):
        self.acc = self.acc * 0



    def _ipython_display_(self):
        print(str(self))



class InteractionField:#описание сил взаимодействия
    def __init__(self, F):
        self.points = []
        self.F = F

    def move_all(self, dt):
        for p in self.points:
            p.move(dt)

    def intensity(self, coord):
        proj = Vector(*[0 for i in range(coord.dim())])
        single_point = Point(Vector(), mass=1.0, q=1.0)
        for p in self.points:
            if coord % p.coords < 0.000001:
                continue
            d = p.coords % coord
            fmod = self.F(single_point, p, d) * (-1)
            proj = proj + (coord - p.coords) / d * fmod
        return proj

    def step(self, dt, h):
        self.clean_acc()
        for p in self.points:
            p.accinc(self.intensity(p.coords) * p.q)
            p.accelerate(dt)
            p.move(dt)
            h = h + dt * 1
            if (p.coords[1]<0):
                p.speed[1]=p.speed[1]*(-1.1)
                p.coords[1]=0
            if p.q > 0:
                p.q = p.q + (h / (p.coords[1] + 0.05)) ** 0.5
            else:
                p.q = p.q - (h / (p.coords[1] + 0.05)) ** 0.5
            for k in self.points:
                if (k.q!=p.q) and (p.coords % k.coords < 0.2):
                    k.q=(p.q+k.q)/2
                    p.q=k.q


    def clean_acc(self):
        for p in self.points:
            p.clean_acc()

    def append(self, *args, **kwargs):
        self.points.append(Point(*args, **kwargs))

    def gather_coords(self):
        return [p.coords for p in self.points]

u = InteractionField(lambda p1, p2, r: (6500 * (-p1.q * p2.q) / (r ** 2 + 0.1)))#кулоновское взаимодейсвие, коэф к другой
for i in range(60):#задаем количество шариков
    u.append(Vector.randvec(2,50,20)+0.1, q=random.random()-0.5)#задаем случайные координаты и заряд

#теперь просто выводим
fig = plt.figure(figsize=[10, 8])
plt.ylim(-5,150)
plt.xlim(-80,110)
phasa = np.arange(0,1000,1)
frames=[]
h = 0
for p in phasa:
    u.step(0.0004, h)
    xd, yd = zip(*u.gather_coords())
    line = plt.scatter(xd,yd,color='r')
    frames.append([line])

animation = ArtistAnimation(
    fig,
    frames,
    interval=15,
    blit=True,
    repeat=True)

plt.show()
print(p.coords)