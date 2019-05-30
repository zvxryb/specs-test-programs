extern crate cgmath;
#[macro_use]
extern crate glium;
extern crate rand;
extern crate rayon;
#[macro_use]
extern crate shred_derive;
extern crate specs;

use cgmath::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use specs::prelude::*;

use cgmath::{Point2, Vector2};
use glium::glutin as glutin;
use rand::distributions as rand_dist;
use specs::shred::{Resource};

use std::time::{Duration, Instant};
use std::f64::consts::PI as PI_F64;

const WORLD_HISTORY: usize = 32;
const VERTEX_BUFFER_SIZE: usize = 1 << 15;

trait AsSeconds<T> {
    fn as_seconds(&self) -> T;
}

impl AsSeconds<f32> for Duration {
    fn as_seconds(&self) -> f32 {
        self.as_nanos() as f32 / 1_000_000_000f32
    }
}

#[derive(Clone, Resource)]
struct Clock {
    last_frame: Instant,
    display: Duration,
    simulation: Duration,
}

impl Clock {
    const TICK_RATE_HZ: u32 = 60;
    const TICK_DELTA_US: u32 = 1_000_000 / Self::TICK_RATE_HZ;
    const TICK_DELTA: f32 = 1f32 / (Self::TICK_RATE_HZ as f32);
    const MIN_DISPLAY_RATE_HZ: u32 = 30;
    const MAX_EXTRAPOLATION_US: u32 = 1_000_000 / Self::MIN_DISPLAY_RATE_HZ;
    const MAX_TICKS: u32 = Self::TICK_RATE_HZ / Self::MIN_DISPLAY_RATE_HZ;

    fn advance(&mut self, tick_counter: &mut u32, real_time: Instant) -> bool {
        let tick_delta = Duration::from_micros(Self::TICK_DELTA_US as u64);
        let real_delta = real_time - self.last_frame;
        if real_delta < tick_delta {
            self.display = self.simulation + real_delta;
            return false;
        }

        if *tick_counter >= Self::MAX_TICKS {
            self.last_frame = real_time;
            self.display = self.simulation + Duration::from_micros(Self::MAX_EXTRAPOLATION_US as u64);
            return false;
        }
        *tick_counter += 1;

        self.last_frame += tick_delta;
        self.simulation += tick_delta;
        true
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self{
            last_frame: Instant::now(),
            display: Default::default(),
            simulation: Default::default(),
        }
    }
}

#[derive(Clone, Default, Resource)]
struct ScreenSize(u32, u32);

#[derive(Clone, Copy)]
struct Position(Point2<f32>);

#[derive(Clone, Copy)]
struct Velocity(Vector2<f32>);

#[derive(Clone, Copy)]
struct Color(f32, f32, f32, f32);

#[derive(Clone)]
struct Lifetime(Duration);

#[derive(Clone, Default)]
struct FirstStage;

impl Component for Position {
    type Storage = specs::VecStorage<Self>;
}

impl Component for Velocity {
    type Storage = specs::VecStorage<Self>;
}

impl Component for Color {
    type Storage = specs::VecStorage<Self>;
}

impl Component for Lifetime {
    type Storage = specs::VecStorage<Self>;
}

impl Component for FirstStage {
    type Storage = specs::NullStorage<Self>;
}

#[derive(Default)]
struct Lifecycle {
    pending: f32
}

impl Lifecycle {
    const SPAWN_RATE: f32 = 50f32;
}

impl<'a> System<'a> for Lifecycle {
    type SystemData = (
        specs::Entities<'a>,
        specs::Read<'a, Clock>,
        specs::WriteStorage<'a, Lifetime>,
        specs::WriteStorage<'a, FirstStage>,
        specs::WriteStorage<'a, Position>,
        specs::WriteStorage<'a, Velocity>,
        specs::WriteStorage<'a, Color>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (entities, clock, mut lifetimes, mut first_stage, mut rs, mut vs, mut colors) = data;

        let dt = Clock::TICK_DELTA;
        let secondary_rs = (&entities, &lifetimes).par_join()
            .flat_map(|(entity, &Lifetime(expires))| {
                if clock.simulation >= expires {
                    let result = if let (Some(_), Some(r)) = (first_stage.get(entity), rs.get(entity)) {
                        Some(*r)
                    } else {
                        None
                    };
                    entities.delete(entity).unwrap();
                    result
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        secondary_rs.into_iter().for_each(|r| {
            let duration_dist = rand_dist::Normal::new(1f64, 0.01f64);
            let velocity_dist = rand_dist::Normal::new(0.1f64, 0.03f64);
            let angle_dist    = rand_dist::Uniform::new(0f64, 2f64 * PI_F64);

            (0..20).into_iter().for_each(|_| {
                let mut rng = rand::thread_rng();
                let duration_us = (1_000_000f64 * duration_dist.sample(&mut rng)) as u64;
                let velocity = velocity_dist.sample(&mut rng) as f32;
                let angle = angle_dist.sample(&mut rng) as f32;
                let v = Vector2::new(angle.cos(), angle.sin()) * velocity;
                entities.build_entity()
                    .with(Lifetime(clock.simulation + Duration::from_micros(duration_us)), &mut lifetimes)
                    .with(r, &mut rs)
                    .with(Velocity(v), &mut vs)
                    .with(Color(1f32, 1f32, 1f32, 1f32), &mut colors)
                    .build();
            });
        });

        let duration_dist = rand_dist::Normal::new(3f64, 0.5f64);
        let pos_dist      = rand_dist::Uniform::new(0f64, 1f64);
        let velocity_dist = rand_dist::Normal::new(0.35f64, 0.01f64);
        let angle_dist    = rand_dist::Normal::new(PI_F64/2f64, 0.01f64 * PI_F64);

        self.pending += dt * Self::SPAWN_RATE;
        let n = self.pending as usize;
        self.pending -= n as f32;
        (0..n).into_iter().for_each(|_| {
            let mut rng = rand::thread_rng();
            let duration_us = (1_000_000f64 * duration_dist.sample(&mut rng)) as u64;
            let velocity = velocity_dist.sample(&mut rng) as f32;
            let angle = angle_dist.sample(&mut rng) as f32;
            let r = Point2::new(pos_dist.sample(&mut rng) as f32, 0f32);
            let v = Vector2::new(angle.cos(), angle.sin()) * velocity;
            entities.build_entity()
                .with(FirstStage::default(), &mut first_stage)
                .with(Lifetime(clock.simulation + Duration::from_micros(duration_us)), &mut lifetimes)
                .with(Position(r), &mut rs)
                .with(Velocity(v), &mut vs)
                .with(Color(1f32, 1f32, 1f32, 1f32), &mut colors)
                .build();
        })
    }
}

#[derive(Default)]
struct Kinematics;

impl<'a> System<'a> for Kinematics {
    type SystemData = (
        specs::WriteStorage<'a, Position>,
        specs::WriteStorage<'a, Velocity>,
    );

    fn run(&mut self, (mut rs, mut vs): Self::SystemData) {
        let dt = Clock::TICK_DELTA;
        (&mut rs, &vs).par_join()
            .for_each(|(Position(r), Velocity(v))| {
                *r += dt * v;
            });
        (&mut vs).par_join()
            .for_each(|Velocity(v)| {
                v.y -= dt * 0.1f32;
            });
    }
}

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}
implement_vertex!(Vertex, position, color);

struct Line<T>(T, T);
struct LineIterator<T>(Line<T>);

impl<T: Send> IntoParallelIterator for Line<T> {
    type Item = T;
    type Iter = LineIterator<T>;

    fn into_par_iter(self) -> LineIterator<T> {
        LineIterator(self)
    }
}

impl<T: Send> ParallelIterator for LineIterator<T> {
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<T>,
    {
        use rayon::iter::plumbing::Folder;
        let Self(Line(a, b)) = self;
        consumer.into_folder()
            .consume(a)
            .consume(b)
            .complete()
    }
}

struct GameState<'a, 'b> {
    events_loop: glutin::EventsLoop,
    display: glium::Display,
    index: usize,
    worlds: Vec<specs::World>,
    dispatcher: Dispatcher<'a, 'b>,
    program: glium::Program,
    vertices: glium::VertexBuffer<Vertex>,
}

impl<'a, 'b> GameState<'a, 'b> {
    fn update(&mut self) {
        self.index = (self.index + 1) % WORLD_HISTORY;

        let (world_src, world_dst) = if self.index == 0 {
            let (head, tail) = self.worlds.split_at_mut(1);
            (tail.last().unwrap(), head.first_mut().unwrap())
        } else {
            let (head, tail) = self.worlds.split_at_mut(self.index);
            (head.last().unwrap(), tail.first_mut().unwrap())
        };

        *world_dst = world_src.clone();

        let real_time = Instant::now();
        let mut tick_count = 0;
        while world_dst.write_resource::<Clock>()
            .advance(&mut tick_count, real_time)
        {
            self.dispatcher.dispatch(&world_dst.res);
            world_dst.maintain();
        }
    }

    fn draw(&mut self) {
        use glium::Surface;

        let mut frame = self.display.draw();
        frame.clear_color_and_depth((0f32, 0f32, 0f32, 1f32), 1f32);

        let worlds: Vec<_> = (0..WORLD_HISTORY).into_iter()
            .map(|i| (WORLD_HISTORY + self.index - i) % WORLD_HISTORY)
            .map(|i| &self.worlds[i])
            .collect();
        let ScreenSize(w, h) = *worlds[0].read_resource::<ScreenSize>();
        let mut brightness = 1.0;
        for i in 0..(WORLD_HISTORY - 1) {
            let vertices: Vec<_> = {
                let world0 = worlds[i];
                let world1 = worlds[i+1];
                let clock0 = world0.read_resource::<Clock>();
                let clock1 = world1.read_resource::<Clock>();
                let dt0 = (clock0.display - clock0.simulation).as_seconds();
                let dt1 = (clock1.display - clock1.simulation).as_seconds();
                let e0 = world0.entities();
                let e1 = world1.entities();
                let r0 = world0.read_storage::<Position>();
                let r1 = world1.read_storage::<Position>();
                let v0 = world0.read_storage::<Velocity>();
                let v1 = world1.read_storage::<Velocity>();
                let c0 = world0.read_storage::<Color>();
                let c1 = world1.read_storage::<Color>();
                (&e0, &e1, &r0, &r1, &v0, &v1, &c0, &c1).par_join()
                    .filter(|(e0, e1, _, _, _, _, _, _)| e0 == e1)
                    .flat_map(|(_, _, &r0, &r1, &v0, &v1, &c0, &c1)| {
                        Line((dt0, r0, v0, c0), (dt1, r1, v1, c1))
                    })
                    .map(|(dt, Position(r), Velocity(v), c)| {
                        Vertex{
                            position: (r + v * dt).into(),
                            color: [
                                brightness * c.0,
                                brightness * c.1,
                                brightness * c.2,
                                c.3]
                        }
                    })
                    .collect()
            };
            let iter = &mut vertices.iter().peekable();
            while let Some(_) = iter.peek() {
                assert!(VERTEX_BUFFER_SIZE % 2 == 0);
                let data: Vec<_> = iter.take(VERTEX_BUFFER_SIZE).cloned().collect();
                let slice = self.vertices.slice(0..data.len()).unwrap();
                slice.write(data.as_slice());

                let params = glium::DrawParameters{
                    blend: glium::Blend{
                        color: glium::BlendingFunction::Addition{
                            source: glium::LinearBlendingFactor::One,
                            destination: glium::LinearBlendingFactor::One,
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let uniforms = uniform!{ screen_size: [w as f32, h as f32] };
                frame.draw(
                    slice,
                    glium::index::NoIndices(glium::index::PrimitiveType::LinesList),
                    &self.program, &uniforms, &params)
                    .expect("failed to draw frame");
            }
            brightness *= 0.8;
        }

        frame.finish().unwrap();
    }

    fn run(&mut self) {
        let mut running = true;
        while running {
            let mut screen_size: Option<ScreenSize> = None;
            self.events_loop.poll_events(|event| {
                use glutin::{Event, WindowEvent};
                match event {
                    Event::WindowEvent{event, ..} => {
                        match event {
                            WindowEvent::CloseRequested => {
                                running = false
                            }
                            WindowEvent::Resized(size) => {
                                screen_size = Some(ScreenSize(
                                    size.width as u32,
                                    size.height as u32))
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            });

            if let Some(size) = screen_size {
                let world = &mut self.worlds[self.index];
                let mut screen_size = world.write_resource::<ScreenSize>();
                *screen_size = size;
            }
            self.update();
            self.draw();
        }
    }
}

fn main() {
    let events_loop = glutin::EventsLoop::new();
    let window_builder = glutin::WindowBuilder::new()
        .with_title("specs multi-world test")
        .with_resizable(true);
    let context = glutin::ContextBuilder::new()
        .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (4, 5)))
        .with_gl_profile(glutin::GlProfile::Core)
        .with_multisampling(4);
    let display = glium::Display::new(window_builder, context, &events_loop)
        .expect("failed to create display");

    let lifecycle = Lifecycle::default();
    let kinematics = Kinematics::default();
    let dispatcher = DispatcherBuilder::new()
        .with(kinematics, "kinematics", &[])
        .with_barrier()
        .with(lifecycle, "lifecycle", &[])
        .build();

    let program = glium::Program::from_source(&display,
        r#"
            #version 450 core

            in vec2 position;
            in vec4 color;

            out vec4 v_color;

            uniform vec2 screen_size;

            void main() {
                v_color = color;
                gl_Position = vec4(2.0 * position * vec2(1.0, screen_size.x / screen_size.y) - 1.0, 0.0, 1.0);
            }
        "#,
        r#"
            #version 450 core

            in vec4 v_color;

            out vec4 f_color;

            void main() {
                f_color = v_color;
            }
        "#,
        None)
        .expect("failed to compile shader");

    let vertices = glium::VertexBuffer::empty_dynamic(&display, VERTEX_BUFFER_SIZE)
        .expect("failed to create vbo");

    let mut state = GameState {
        events_loop,
        display,
        index: 0usize,
        worlds: (0..WORLD_HISTORY)
            .map(|_| {
                let mut world = specs::World::new();
                world.add_resource(Clock::default());
                world.add_resource(ScreenSize::default());
                world.register::<Position>();
                world.register::<Velocity>();
                world.register::<Color>();
                world.register::<Lifetime>();
                world.register::<FirstStage>();
                world
            })
            .collect(),
        dispatcher,
        program,
        vertices,
    };

    state.run();
}