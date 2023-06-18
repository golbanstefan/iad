create table if not exists autori
(
    id         int auto_increment
        primary key,
    nume       varchar(255) not null,
    profil_url varchar(255) not null
);

create table if not exists proprietati_nenormalizate
(
    id                varchar(255)   not null,
    titlu             varchar(255)   not null,
    pret              decimal(10, 2) not null,
    actualizat_la     datetime       not null,
    tipul             varchar(255)   not null,
    nume_autor        varchar(255)   not null,
    profil_autor      varchar(255)   not null,
    regiune           varchar(255)   not null,
    tip_locuinta      varchar(255)   not null,
    suprafata_totala  decimal(10, 2) not null,
    stare_proprietate varchar(255)   not null,
    nr_camere         int            not null,
    balcon            int            not null,
    etaj              int            not null,
    nr_etaje          int            not null,
    lon               decimal(10, 8) not null,
    lat               decimal(10, 8) not null,
    data_modificare   timestamp      not null
);

create table if not exists stari_proprietati
(
    id    int auto_increment
        primary key,
    stare varchar(255) not null
);

create table if not exists tip_locuinte
(
    id  int auto_increment
        primary key,
    tip varchar(255) not null
);

create table if not exists proprietati
(
    id                   varchar(255)   not null
        primary key,
    titlu                varchar(255)   not null,
    pret                 decimal(10, 2) not null,
    actualizat_la        datetime       null,
    tipul                varchar(255)   not null,
    id_autor             int            null,
    regiune              varchar(255)   not null,
    id_tip_locuinta      int            null,
    suprafata_totala     decimal(10, 2) not null,
    id_stare_proprietate int            null,
    nr_camere            int            not null,
    balcon               int            not null,
    etaj                 int            not null,
    nr_etaje             int            not null,
    lon                  decimal(10, 8) not null,
    lat                  decimal(10, 8) not null,
    constraint proprietati_ibfk_1
        foreign key (id_autor) references autori (id),
    constraint proprietati_ibfk_2
        foreign key (id_tip_locuinta) references tip_locuinte (id),
    constraint proprietati_ibfk_3
        foreign key (id_stare_proprietate) references stari_proprietati (id)
);

create index id_autor
    on proprietati (id_autor);

create index id_stare_proprietate
    on proprietati (id_stare_proprietate);

create index id_tip_locuinta
    on proprietati (id_tip_locuinta);

create definer = root@`%` trigger update_proprietati_nenormalizate_after_insert
    after insert
    on proprietati
    for each row
BEGIN
    INSERT INTO proprietati_nenormalizate (
        id, titlu, pret, actualizat_la, tipul, nume_autor, profil_autor, regiune, tip_locuinta,
        suprafata_totala, stare_proprietate, nr_camere, balcon, etaj, nr_etaje, lon, lat, data_modificare
    )
    SELECT
        p.id, p.titlu, p.pret, p.actualizat_la, p.tipul, a.nume, a.profil_url, p.regiune, t.tip,
        p.suprafata_totala, s.stare, p.nr_camere, p.balcon, p.etaj, p.nr_etaje, p.lon, p.lat, NOW()
    FROM
        proprietati p
        JOIN autori a ON p.id_autor = a.id
        JOIN tip_locuinte t ON p.id_tip_locuinta = t.id
        JOIN stari_proprietati s ON p.id_stare_proprietate = s.id
    WHERE p.id = NEW.id;
END;

